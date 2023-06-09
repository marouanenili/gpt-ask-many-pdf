from sklearn.metrics.pairwise import cosine_distances
import pdf
import ai
import re
import streamlit as st
import pinecone

def use_key(api_key):
	ai.use_key(api_key)

def query_by_vector(vector, index, limit=None):
	"return (ids, distances and texts) sorted by cosine distance"
	vectors = index['vectors']
	texts = index['texts']
	#
	sim = cosine_distances([vector], vectors)[0]
	#
	id_dist_list = list(enumerate(sim))
	id_dist_list.sort(key=lambda x:x[1])
	id_list   = [x[0] for x in id_dist_list][:limit]
	dist_list = [x[1] for x in id_dist_list][:limit]
	text_list = [texts[x] for x in id_list] if texts else ['ERROR']*len(id_list)
	return id_list, dist_list, text_list

def get_vectors(text_list, pg=None):
	"transform texts into embedding vectors"
	vectors = []
	for i,text in enumerate(text_list):
		resp = ai.embedding(text)
		v = resp['vector']
		vectors += [v]
		print("vectorizing :"+str(i)+"/"+str(len(text_list)))
		if pg:
			pg.progress((i+1)/len(text_list))
	return vectors

def index_file(f, fix_text=False, frag_size=0, pg=None):
	"return vector index (dictionary) for a given PDF file"
	pages = pdf.pdf_to_pages(f)
	if fix_text:
		for i in range(len(pages)):
			print("fixing page :"+str(i)+"/"+str(len(pages)))
			pages[i] = fix_text_problems(pages[i], pg)
	texts = split_pages_into_fragments(pages, frag_size)
	vectors = get_vectors(texts, pg)
	summary_prompt = f"{texts[0]}\n\nDescribe the document from which the fragment is extracted. Omit any details.\n\n" # TODO: move to prompts.py
	summary = ai.old_complete(summary_prompt)
	out = {}
	out['size']    = len(texts)
	out['texts']   = texts
	out['pages']   = pages
	out['vectors'] = vectors
	out['summary'] = summary['text']
	return out

def split_pages_into_fragments(pages, frag_size):
	"split pages (list of texts) into smaller fragments (list of texts)"
	page_offset = [0]
	for p,page in enumerate(pages):
		page_offset += [page_offset[-1]+len(page)+1]
	# TODO: del page_offset[-1] ???
	if frag_size:
		text = ' '.join(pages)
		return text_to_fragments(text, frag_size, page_offset)
	else:
		return pages

def text_to_fragments(text, size, page_offset):
	"split single text into smaller fragments (list of texts)"
	if size and len(text)>size:
		out = []
		pos = 0
		page = 1
		p_off = page_offset.copy()[1:]
		eos = find_eos(text)
		if len(text) not in eos:
			eos += [len(text)]
		for i in range(len(eos)):
			if eos[i]-pos>size:
				text_fragment = f'PAGE({page}):\n'+text[pos:eos[i]]
				out += [text_fragment]
				pos = eos[i]
				if eos[i]>p_off[0]:
					page += 1
					del p_off[0]
		# ugly: last iter
		text_fragment = f'PAGE({page}):\n'+text[pos:eos[i]]
		out += [text_fragment]
		#
		out = [x for x in out if x]
		return out
	else:
		return [text]

def find_eos(text):
	"return list of all end-of-sentence offsets"
	return [x.span()[1] for x in re.finditer('[.!?]\s+',text)]

###############################################################################

def fix_text_problems(text, pg=None):
	"fix common text problems"
	text = re.sub('\s+[-]\s+','',text) # word continuation in the next line
	return text


def query(selected_options,text, index, task=None, temperature=0.0, max_frags=1, hyde=False, hyde_prompt=None, limit=None):
	"get dictionary with the answer for the given question (text)."
	out = {}
	print(selected_options)
	if hyde:
		out['hyde'] = hypotetical_answer(text, index, hyde_prompt=hyde_prompt, temperature=temperature)
	
	# RANK FRAGMENTS
	if hyde:
		resp = ai.embedding(out['hyde']['text'])
	else:
		resp = ai.embedding(text)
	v = resp['vector']
	id_list, dist_list, text_list = query_by_vector(v, index, limit=limit)
	
	# BUILD PROMPT
	
	# select fragments
	N_BEFORE = 1 # TODO: param
	N_AFTER =  1 # TODO: param
	selected = {} # text id -> rank
	for rank,id in enumerate(id_list):
		for x in range(id-N_BEFORE, id+1+N_AFTER):
			if x not in selected and x>=0 and x<index['size']:
				selected[x] = rank
	selected2 = [(id,rank) for id,rank in selected.items()]
	selected2.sort(key=lambda x:(x[1],x[0]))
	
	# build context
	SEPARATOR = '\n---\n'
	context = ''
	context_len = 0
	frag_list = []
	for id,rank in selected2:
		frag = index['texts'][id]
		frag_len = ai.get_token_count(frag)
		if context_len+frag_len <= 3000: # TODO: remove hardcode
			context += SEPARATOR + frag # add separator and text fragment
			frag_list += [frag]
			context_len = ai.get_token_count(context)
	out['context_len'] = context_len
	prompt = f"""
		{task or 'Task: Answer question based on context.'}
		
		Context:
		{context}
		
		Question: {text}
		
		Answer:""" # TODO: move to prompts.py
	#Task = "Answer the question truthfully based on the text below. Include verbatim quote and a comment where to find it in the text (page and section number). After the quote write a step by step explanation. Use bullet points. Create a one sentence summary of the preceding output."

	message = [
		{"role":"system", "content": "answer ONLY based on the context "+ task },
		{"role":"system", "content": "Context" + context},
		{"role":"user", "content": "Question: "+text},
	]
	
	# GET ANSWER
	resp2 = ai.complete(prompt, temperature=temperature,messages = message)
	answer = resp2['text']
	usage = resp2['usage']
	
	# OUTPUT
	out['id_list'] = id_list
	out['dist_list'] = dist_list
	out['selected'] = selected
	out['selected2'] = selected2
	out['frag`_list'] = frag_list
	#out['query.vector'] = resp['vector']
	out['usage'] = usage
	out['prompt'] = prompt
	out['text'] = answer
	return out

def get_response_vectors(options,text):
	index_name = 'regulations'
	pinecone.init(
		api_key=st.secrets['API_KEY_pinecone'],
		environment="us-west1-gcp"
	)
	index = pinecone.Index(index_name)
	v = get_vectors([text])
	responses = []
	dict_namespaces = {
		"(Sofars) Purpose, Authority, Administration, Agency acquisition regulations, deviations from the FAR": "SOFARS Full",
		"(TRANSFARS) policies and procedures for USTRANSCOM contracting officers": "TRANSFARS",
		"Air Force Federal Acquisition Regulation Supplement": "AFFARS",
		"Air Force Federal Acquisition Regulation Supplement Full": "AFARS_FULL",
		"Broadcasting Board of Governors Acquisition Regulation Supplement": "CFR-2021-title48-vol6-chap19",
		"Defense Federal Acquisition Regulation": "DFARS",
		"Department of Justice": "CFR-2021-title48-vol6-chap28",
		"Department of Agriculture's Acquisition Regulation System": "CFR-2021-title48-vol4-chap4",
		"Department of Commerce Acquisition Regulations System": "CFR-2021-title48-vol5-chap13",
		"Department of Education Acquisition Regulation": "CFR-2021-title48-vol7-chap34",
		"Department of Energy's Federal Acquisition Regulations System": "CFR-2021-title48-vol5-chap9",
		"Department of Homeland Security's Homeland Security Acquisition Regulation (HSAR)":"CFR-2021-title48-vol7-chap30",
		"Department of Housing and Urban Development's Federal Acquisition Regulation System": "CFR-2021-title48-vol6-chap24",
		"Department of Labor Acquisition Regulation System": "CFR-2021-title48-vol7-chap29",
		"Department of State Acquisition Regulations System": "CFR-2021-title48-vol4-chap6",
		"Department of the Interior Acquisition Regulation System": "CFR-2021-title48-vol5-chap14",
		"Department of the Treasury Acquisition Regulation (DTAR) System": "CFR-2021-title48-vol5-chap10",
		"Department of Transportation": "CFR-2021-title48-vol5-chap12",
		"Department of Veterans Affairs Acquisition Regulation System": "vaar",
		"DEPARTMENT OF VETERANS  AFFAIRS": "CFR-2021-title48-vol5-chap8",
		"Environmental Protection Agency's regulations": "CFR-2021-title48-vol6-chap15",
		"FAR_Federal Acquisition Regulation for Fiscal Year 2019": "FAR",
		"Federal Aquisition System (DARS)": "DARS_Full",
		"Federal Aquisition System (DLAD)": "DLAD_Full",
		"GSA Acquisition Manual (GSAM)": "GSAM",
		"HHS Acquisition Regulation System": "CFR-2021-title48-vol4-chap3",
		"Nuclear Regulatory Commission Acquisition Regulation System": "CFR-2021-title48-vol6-chap20",
		"Office of Personnel Management Federal Employees Health Benefits Acquisition": "CFR-2021-title48-vol6-chap16",
		"Office of Personnel  Management, Federal Employees Group  Life Insurance Federal Acquisition  Regulation ": "CFR-2021-title48-vol6-chap21",
		"Regulations for federal government procurement": "NMCARS_Full",
		"Rules and regulations for government procurement": "CFR-2021-title48-vol6-chap18",
		"The Federal Acquisition Regulation (FAR) Volume III-P Arts 201 to 253, issued for Fiscal Year 2020": "DFARSPGI",

	}
	names_spaces = [dict_namespaces[option] for option in options]
	for name_space in names_spaces:
		print(name_space)

		response = index.query(vector=v,top_k=1,include_values=True,namespace=name_space,include_metadata=True)
		if len(response['matches']) == 0:
			continue
		elif response['matches'][0]['score'] > 0.5:
			responses.append((response,name_space))
	if len(responses) == 0:
		return []
	response_final = responses[0]
	for response in responses:
		if response[0]['matches'][0]['score'] > response_final[0]['matches'][0]['score']:
			response_final = response
	return response_final

def query2(response_final_and_regulation,text, temperature=0.0, max_frags=1, hyde=False, hyde_prompt=None, limit=None):
	dict_namespaces = {
		"(Sofars) Purpose, Authority, Administration, Agency acquisition regulations, deviations from the FAR": "SOFARS Full",
		"(TRANSFARS) policies and procedures for USTRANSCOM contracting officers": "TRANSFARS",
		"Air Force Federal Acquisition Regulation Supplement": "AFFARS",
		"Air Force Federal Acquisition Regulation Supplement Full": "AFARS_FULL",
		"Broadcasting Board of Governors Acquisition Regulation Supplement": "CFR-2021-title48-vol6-chap19",
		"Defense Federal Acquisition Regulation": "DFARS",
		"Department of Justice": "CFR-2021-title48-vol6-chap28",
		"Department of Agriculture's Acquisition Regulation System": "CFR-2021-title48-vol4-chap4",
		"Department of Commerce Acquisition Regulations System": "CFR-2021-title48-vol5-chap13",
		"Department of Education Acquisition Regulation": "CFR-2021-title48-vol7-chap34",
		"Department of Energy's Federal Acquisition Regulations System": "CFR-2021-title48-vol5-chap9",
		"Department of Homeland Security's Homeland Security Acquisition Regulation (HSAR)": "CFR-2021-title48-vol7-chap30",
		"Department of Housing and Urban Development's Federal Acquisition Regulation System": "CFR-2021-title48-vol6-chap24",
		"Department of Labor Acquisition Regulation System": "CFR-2021-title48-vol7-chap29",
		"Department of State Acquisition Regulations System": "CFR-2021-title48-vol4-chap6",
		"Department of the Interior Acquisition Regulation System": "CFR-2021-title48-vol5-chap14",
		"Department of the Treasury Acquisition Regulation (DTAR) System": "CFR-2021-title48-vol5-chap10",
		"Department of Transportation": "CFR-2021-title48-vol5-chap12",
		"Department of Veterans Affairs Acquisition Regulation System": "vaar",
		"DEPARTMENT OF VETERANS  AFFAIRS": "CFR-2021-title48-vol5-chap8",
		"Environmental Protection Agency's regulations": "CFR-2021-title48-vol6-chap15",
		"FAR_Federal Acquisition Regulation for Fiscal Year 2019": "FAR",
		"Federal Aquisition System (DARS)": "DARS_Full",
		"Federal Aquisition System (DLAD)": "DLAD_Full",
		"GSA Acquisition Manual (GSAM)": "GSAM",
		"HHS Acquisition Regulation System": "CFR-2021-title48-vol4-chap3",
		"Nuclear Regulatory Commission Acquisition Regulation System": "CFR-2021-title48-vol6-chap20",
		"Office of Personnel Management Federal Employees Health Benefits Acquisition": "CFR-2021-title48-vol6-chap16",
		"Office of Personnel  Management, Federal Employees Group  Life Insurance Federal Acquisition  Regulation ": "CFR-2021-title48-vol6-chap21",
		"Regulations for federal government procurement": "NMCARS_Full",
		"Rules and regulations for government procurement": "CFR-2021-title48-vol6-chap18",
		"The Federal Acquisition Regulation (FAR) Volume III-P Arts 201 to 253, issued for Fiscal Year 2020": "DFARSPGI",

	}
	new_dict = {v: k for k, v in dict_namespaces.items()}
	task = "Answer the question truthfully based on the text below. Include verbatim quote and a comment where to find it in the text (page and section number). After the quote write a step by step explanation. Use bullet points. Create a one sentence summary of the preceding output."
	import pickle
	"get dictionary with the answer for the given question (text)."
	response_final = response_final_and_regulation[0]
	regulation = response_final_and_regulation[1]
	id = int(response_final['matches'][0]['metadata']['page'])
	index_name = new_dict[response_final['namespace']]
	with open("src/pkl/"+index_name+".pkl", "rb") as f:
		index = pickle.load(f)
	out = {}
	selected = []
	if id-1>0:
		selected.append(id-1)
	selected.append(id)
	if id+1<len(index['texts']):
		selected.append(id+1)
	# build context
	SEPARATOR = '\n---\n'
	context = ''
	context_len = 0
	frag_list = []
	for id in selected:
		frag = index['texts'][id]
		frag_len = ai.get_token_count(frag)
		if context_len + frag_len <= 3000:  # TODO: remove hardcode
			context += SEPARATOR + frag  # add separator and text fragment
			frag_list += [frag]
			context_len = ai.get_token_count(context)
	out['context_len'] = context_len
	prompt = f"""
		{task or 'Task: Answer question based on context.'}

		Context:
		{context}

		Question: {text}

		Answer:"""  # TODO: move to prompts.py
	# Task = "Answer the question truthfully based on the text below. Include verbatim quote and a comment where to find it in the text (page and section number). After the quote write a step by step explanation. Use bullet points. Create a one sentence summary of the preceding output."

	message = [
		{"role": "system", "content": "answer ONLY based on the context " + task},
		{"role": "system", "content": "Context" + context},
		{"role": "user", "content": "Question: " + text},
	]

	# GET ANSWER
	resp2 = ai.complete(prompt, temperature=temperature, messages=message)
	answer = resp2['text']
	usage = resp2['usage']

	# OUTPUT
	out['selected'] = selected
	out['regulation'] = regulation

	out['frag`_list'] = frag_list
	# out['query.vector'] = resp['vector']
	out['usage'] = usage
	out['prompt'] = prompt
	out['text'] = answer
	return out


def hypotetical_answer(text, index, hyde_prompt=None, temperature=0.0):
	"get hypotethical answer for the question (text)"
	hyde_prompt = hyde_prompt or 'Write document that answers the question.'
	prompt = f"""
	{hyde_prompt}
	Question: "{text}"
	Document:""" # TODO: move to prompts.py
	resp = ai.complete(prompt, temperature=temperature)
	return resp


if __name__=="__main__":
	print(text_to_fragments("to jest. test tego. programu", size=3, page_offset=[0,5,10,15,20]))
	
