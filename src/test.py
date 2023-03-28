import os
import pickle

folder_path = 'pkl'
paths =[]
# Loop through all the files in the folder
for filename in os.listdir(folder_path):
    # Check if the current file is a file or directory
    if os.path.isfile(os.path.join(folder_path, filename)):
        paths.append( folder_path+'/'+filename)
for path in paths:
    with open(path, "rb") as f:
        out = pickle.load(f)
    #creating list to send to pinecone with meta data
    print(path +" : "+out['summary'])