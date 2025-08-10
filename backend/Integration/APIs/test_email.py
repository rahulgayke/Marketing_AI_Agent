"""
This script evaluates the performance of two models on a set of email responses
using BLEU, ROUGE, and BERTScore metrics. It assumes that the model outputs and references are provided as lists of strings.
"""
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import score as bert_score
from dotenv import load_dotenv
load_dotenv()   

llm_mistral = HuggingFaceEndpoint(
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    task = "text-generation"
)
llm_llama3 = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B",
    task = "text-generation"
)
llm_dsr1 = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-R1",
    task = "text-generation"
)

model_mistral = ChatHuggingFace(llm = llm_mistral)
model_llama3 = ChatHuggingFace(llm = llm_llama3)
model_dsr1 = ChatHuggingFace(llm = llm_dsr1)

recipient_name = "Bharat Kamble"
sender_name = "Rahul Gayke"
sample_product_detail_list = [{
    "product_name": "Real Estate Property",
    "product_description": "A luxurious real estate property that offers stunning views and modern amenities."
},
{
    "product_name": "Smartphone",
    "product_description": "A latest model smartphone with advanced features and sleek design."
},
{
    "product_name": "Fitness Tracker",
    "product_description": "A wearable fitness tracker that monitors heart rate, steps, and calories burned."
},
{
    "product_name": "Online Course",
    "product_description": "An interactive online course that helps you learn new skills at your own pace."
},
{
    "product_name": "Subscription Box",
    "product_description": "A monthly subscription box filled with curated products tailored to your interests."
},
{    
    "product_name": "Eco-Friendly Product",
    "product_description": "A sustainable product made from recycled materials."
},
{
    "product_name": "Luxury Watch",
    "product_description": "A high-end luxury watch that combines elegance and functionality."
},
{
    "product_name": "Gourmet Food Basket",
    "product_description": "A selection of high-quality gourmet foods, including artisanal cheeses, cured meats, and fine chocolates."
},
{
    "product_name": "Travel Package",
    "product_description": "An all-inclusive travel package that offers a unique and memorable experience."
},
{
    "product_name": "Home Decor Item",
    "product_description": "A stylish home decor item that adds a touch of elegance to any room."
}
]

reference_results = []
result_list_mistral = []
result_list_llama3 = []
result_list_dsr1 = []
for sample_product_detail in sample_product_detail_list:
    # Draft the email using the model
    prompt = (f"""
        Draft a professional and personalized marketing email addressed to {recipient_name}:
        The email should promote the following product:
        Product Name: {sample_product_detail["product_name"]}
        Product Description: {sample_product_detail["product_description"]}
        Make sure to include a friendly greeting and a closing signature with the sender's name.
        sender_name: {sender_name}
        """)
    
    # Generate email using the models
    result_mistral = model_mistral.invoke(prompt)
    result_llama3 = model_llama3.invoke(prompt)
    result_dsr1 = model_dsr1.invoke(prompt)

    # Append the results to the respective lists
    result_list_mistral.append(result_mistral)
    result_list_llama3.append(result_llama3)
    result_list_dsr1.append(result_dsr1)

# BLEU Score (Bilingual Evaluation Understudy)
# Checks how many n-grams are matching.
def calculate_bleu(candidate, reference):
    from nltk import word_tokenize
    return sentence_bleu([word_tokenize(reference)], word_tokenize(candidate))

bleu_mistral = [calculate_bleu(mistral, r) for mistral, r in zip(result_list_mistral, reference_results)]
bleu_llama3 = [calculate_bleu(llama3, r) for llama3, r in zip(result_list_llama3, reference_results)]
bleu_dsr1 = [calculate_bleu(dsr1, r) for dsr1, r in zip(result_list_dsr1, reference_results)]

# ROUGE Score (Recall-Oriented Understudy for Gisting Evaluation)
# Higher ROUGE-L means the generated text has more overlap
rouge = Rouge()
rouge_mistral = [rouge.get_scores(mistral, r)[0] for mistral, r in zip(result_list_mistral, reference_results)]
rouge_llama3 = [rouge.get_scores(llama3, r)[0] for llama3, r in zip(result_list_llama3, reference_results)]
rouge_dsr1 = [rouge.get_scores(dsr1, r)[0] for dsr1, r in zip(result_list_dsr1, reference_results)]

# BERTScore
# Closer to 1.0 means the generation is semantically very similar to the human reference, even if the wording differs.
P_mistral, R_mistral, F1_mistral = bert_score(result_list_mistral, reference_results, lang="en")
P_llama3, R_llama3, F1_llama3 = bert_score(result_list_llama3, reference_results, lang="en")
P_dsr1, R_dsr1, F1_dsr1 = bert_score(result_list_dsr1, reference_results, lang="en")

# Report Means
print("Model MMistral BLEU:", sum(bleu_mistral)/len(bleu_mistral))
print("Model Llama3 BLEU:", sum(bleu_llama3)/len(bleu_llama3))
print("Model DSR1 BLEU:", sum(bleu_dsr1)/len(bleu_dsr1))
print("Model Mistral ROUGE-L:", sum([r['rouge-l']['f'] for r in rouge_mistral])/len(rouge_mistral))
print("Model Llama3 ROUGE-L:", sum([r['rouge-l']['f'] for r in rouge_llama3])/len(rouge_llama3))
print("Model DSR1 ROUGE-L:", sum([r['rouge-l']['f'] for r in rouge_dsr1])/len(rouge_dsr1))
print("Model Mistral BERTScore F1:", F1_mistral.mean().item())
print("Model Llama3 BERTScore F1:", F1_llama3.mean().item())
print("Model DSR1 BERTScore F1:", F1_dsr1.mean().item())
