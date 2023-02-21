#open file to read
import torch
with open("input.txt","r",encoding="utf-8") as f:
    text = f.read()

#print("length of dataset in characters: " , len(text))

#number of possible characters
chars = sorted(list(set(text)))
vocab_size = len(chars) 
#print("".join(chars)) #appends (join) to "" all the characters on the list chars
#print(vocab_size)

#tokenize input text (convert raw string to integers)
stoi = {ch:i for i, ch in enumerate(chars)}  #string to int
itos = { i:ch for i, ch in enumerate(chars)} #int to string
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

#print(encode("hii there"))
#print(decode(encode("hii there")))

data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)
#print(data[:1000]) #print all the first tokenized characters
#the first 1000 characters of the text as a list of integers 

#split the data into validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target is: {target}")
