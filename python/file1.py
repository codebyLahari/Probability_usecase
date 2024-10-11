
file_path=r"C:\Users\lahar\Gen-AI\chunk.txt"
chunk_file_path=r"C:\Users\lahar\Gen-AI\chunk_files"
with open(file_path , "r") as f:
    content=f.read()

sentences = content.split("\n") 

num_files = 10
sentences_per_file = len(sentences) // num_files 

for i in range(num_files):
    chunk = sentences[i * sentences_per_file: (i + 1) * sentences_per_file]
    with open(f"chunk_{i+1}.txt", "w") as chunk_file_path:
        chunk_file_path.write("\n".join(chunk))
         


