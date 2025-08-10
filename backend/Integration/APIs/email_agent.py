
from transformers import pipeline, AutoTokenizer
from langchain_core.prompts import PromptTemplate
import APIs.constants as C
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage
from email_validator import validate_email, EmailNotValidError
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

class EmailAgent():
    """
    Handles generating personalized marketing emails and sending them.
    """
    def __init__(self):
        # Load the Hugging Face model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(C.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(C.MODEL_NAME)
        self.sender_name = C.SENDER_NAME
        self.sender_email = C.SENDER_EMAIL
        self.sender_password = C.SENDER_PASSWORD

    def draft_marketing_email(self, product_name: str, product_description: str, recipient_name: str) -> str:
        """
        """
        # Text generation pipeline
        # generator = pipeline(
        #     task='text-generation', 
        #     model=self.model,
        #     tokenizer=self.tokenizer, 
        #     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # )
        llm = HuggingFaceEndpoint(
            repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1",
            task = "text-generation"
        )
        model = ChatHuggingFace(llm = llm)
        print(f"Model generated successfully: {model}")

        # Prompt for the LLM
        # prompt_template = PromptTemplate(
        #     template="""Draft a professional and personalized marketing email addressed to {recipient_name}:\n
        #     The email should promote the following product:\n
        #     Product Name: {product_name}\n
        #     Product Description: {product_description}\n
        #     The email should be engaging, concise, and include a call to action.\n
        #     Make sure to include a friendly greeting and a closing signature with the sender's name.\n
        #     sender_name: {sender_name}\n\n

        #     Email:
        #     """,
        #     input_variables=['recipient_name','product_name', 'product_description', 'sender_name']
        # )
        # print(f"Prompt template: {prompt_template}")

        # prompt = prompt_template.invoke({
        #     'recipient_name': recipient_name,
        #     'product_name': product_name,
        #     'product_description': product_description,
        #     'sender_name': self.sender_name
        # })

        prompt = (f"""
        Draft a professional and personalized marketing email addressed to {recipient_name}:
        The email should promote the following product:
        Product Name: {product_name}
        Product Description: {product_description}
        Make sure to include a friendly greeting and a closing signature with the sender's name.
        sender_name: {self.sender_name}
        """)
        
        print(f"Prompt for generator: {prompt}")

        # result = generator(
        #     prompt.text, 
        #     max_length=256, 
        #     do_sample=False
        # )
        result = model.invoke(prompt)
        print(f"Generator response: {result.content}")
        # Remove prompt from result, keep only generated email
        # email_body = result[0]['generated_text'].replace(prompt, "").strip()
        
        # Add sender's signature at the bottom
        # signature = f"\n\nBest regards,\n{sender_name}"

        return result.content

    def send_marketing_email(self, receiver_email: str, email_subject: str, email_body: str):

        """
        Send a marketing email using the provided details.
        """
        # Validate email address
        try:
            validate_email(receiver_email)
            validate_email(self.sender_email)
        except EmailNotValidError as e:
            print(f"Invalid email address: {e}")
            return
        
        msg = EmailMessage()
        msg['Subject'] = email_subject
        msg['From'] = self.sender_email
        msg['To'] = receiver_email
        msg.set_content(email_body)

        # Send via SMTP
        try:
            with smtplib.SMTP(C.SMTP_SERVER, C.SMTP_PORT) as server:
                if C.SMTP_USE_TLS:
                    server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            print("Email sent successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to send email: {e}")

    # Use below code if you need to use a single call for a email task and do the changes to app.py accordingly
    # def agent_call(event, context):
    #     # Get user input (replace with your method of input)
    #     product_info = event.get("product_desc", {})
    #     recipient_name = event.get("recipient_name", "Valued Customer")
    #     sender_name = event.get("sender_name", "Your Company")

    #     # Draft the marketing email using the AI model
    #     generated_email = self.draft_marketing_email(product_info, recipient_name, sender_name)
    #     print("Generated Email:\n", generated_email)

    #     return { "generated_email": json.dumps(generated_email)}

