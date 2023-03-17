__version__ = "Beta"

import base64

app_name = "FARWizard - Ask the FAR"


# BOILERPLATE
import csv
import streamlit as st
import pandas as pd
import time

api_key = ""


st.set_page_config(layout='centered', page_title=f'{app_name} {__version__}',initial_sidebar_state="collapsed")
ss = st.session_state
import css
st.write(f'<style>{css.v1}</style>', unsafe_allow_html=True)
header1 = st.empty() # for errors / messages
header2 = st.empty() # for errors / messages
header3 = st.empty() # for errors / messages

# IMPORTS

import prompts
import model

# COMPONENTS

def ui_spacer(n=2, line=False, next_n=0):
	for _ in range(n):
		st.write('')
	if line:
		st.tabs([' '])
	for _ in range(next_n):
		st.write('')






def ui_question():
	st.title('First Came the FAR, Then Came FARGPT')
	st.header('Why FARGPT and why not ChatGPT?')
	st.caption('FARGPT is built to semantically search the FAR and provide answers and references whereas ChatGPT searches its entire collection of text to find answers which may not be related to the FAR at all.')
	st.caption('FARGPT searches the FAR PDF located on Acquisition.gov: https://www.acquisition.gov/sites/default/files/current/far/pdf/FAR.pdf')
	st.caption('_*For academic & resarch purposes only* - Patent Pending_')
	st.caption('_Questions, Comments, Concerns? -> Support@FARWizard.com_')
	st.write('## Ask the FAR a Question!')
	st.text_area('question', key='question', height=100, placeholder='Enter question here', help='', label_visibility="collapsed", disabled=False)

# REF: Hypotetical Document Embeddings

def ui_output():
	output = ss.get('output','')
	st.markdown(output)

def b_ask():
	disabled = check_disabled()
	if st.button('Submit', disabled=disabled, type='primary'):
		text = ss.get('question','')
		hyde_prompt = ss.get('hyde_prompt')
		if ss.get('use_hyde_summary'):
			summary = ss['index']['summary']
			hyde_prompt += f" Context: {summary}\n\n"

		with st.spinner('preparing answer'):
			vector = model.get_response_vectors(ss.selected_options,text)
			resp = model.query2(vector, text)
		q = text.strip() +' ('+resp['regulation']+")"
		a = resp['text']  #.strip()
		output_add(q,a)
def check_disabled():
	if ss.selected_options is None:
		return True
	elif ss.selected_options == []:
		return True
	return False
def b_clear():
	if st.button('clear output'):
		ss['output'] = ''



def b_reload():
	if st.button('reload prompts'):
		import importlib
		importlib.reload(prompts)


def write_in_csv(q,a):
	file_path = "QAcsv.csv"
	with open(file_path, 'a', newline='\n') as file:
		writer = csv.writer(file)
		writer.writerow([str(q), str(a)])
	file.close()


def output_add(q, a):
	write_in_csv(q, a)
	ss['output'] = ''
	t = st.empty()
	if 'output' not in ss:
		ss['output'] = ""
	new = f'#### {q}\n\n'
	ss['output']=ss['output']+new
	for i in a:
		
		ss['output'] = ss['output'] + i
		t.write(ss['output'])
		time.sleep(0.01)



def select_pdf():
	# %%
	import os

	folder_path = 'pdf'
	options_list = []
	# Loop through all the files in the folder
	for filename in os.listdir(folder_path):
		# Check if the current file is a file or directory
		if os.path.isfile(os.path.join(folder_path, filename)):
			options_list.append(filename[:-4])
	option = st.multiselect('Select three reglementations:',
							options=options_list,max_selections=3)

	if 'selected_options' not in ss:
		st.session_state.selected_options = option
	if option != st.session_state.selected_options:
		st.session_state.selected_options = option

# LAYOUT



Task = "Answer the question truthfully based on the text below. Include verbatim quote and a comment where to find it in the text (page and section number). After the quote write a step by step explanation. Use - as bullet points. Create a one sentence summary of the preceding output."
api_key = st.secrets['API_KEY']
model.use_key(api_key)
secret_key = st.secrets['password']




def page2():

	# Load CSV file into a pandas DataFrame
	df = pd.read_csv('QAcsv.csv')

	# Display the DataFrame in Streamlit
	st.write(df)
	csv = df.to_csv(index=False)
	b64 = base64.b64encode(csv.encode()).decode()

	href = f'<a href="data:file/csv;base64,{b64}" download="QAcsv.csv">Download CSV file</a>'
	st.markdown(href, unsafe_allow_html=True)

timer = 1
def app():
	global timer
	st.sidebar.expander("Navigation", expanded=False)
	selection = st.sidebar.radio("", ["Public", "Admin"], index=0)
	if selection == "Admin":
		provided_key = st.text_input("Enter the secret key to access this page:",type="password")
		if provided_key != secret_key:
			timer += 1
			time.sleep(timer)
			st.write("Incorrect key. You do not have access to this page.")
			return
	timer = 1
	# Run the selected page
	if selection == "Public":
		select_pdf()
		ui_question()
		b_ask()
	elif selection == "Admin":
		page2()
app()

