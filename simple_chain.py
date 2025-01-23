import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.chains import LLMChain 
from langchain.prompts import PromptTemplate

def load_llm(model_name="openai-community/gpt2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=False).to(device)
    return model, tokenizer

def create_prompt(template):
    return PromptTemplate(template=template, input_variables=["question"])

def create_simple_chain(prompt_template, model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def llm_chain(question):
        inputs = tokenizer(prompt_template.format(question=question), return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_p=0.9
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return llm_chain

if __name__ == "__main__":
    template = "Question: {question}\nAnswer:"
    
    model_file = "openai-community/gpt2"
    
    model, tokenizer = load_llm(model_file)
    prompt = create_prompt(template)
    llm_chain = create_simple_chain(template, model, tokenizer)
    
    question = "How many hours a day?"
    response = llm_chain(question)
    print(response)