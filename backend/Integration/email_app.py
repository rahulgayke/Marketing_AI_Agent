
from flask import Flask, app, jsonify
from APIs.email_agent import EmailAgent

# app = Flask(__name__)

# @app.route('/draft_and_send', methods=['POST'])
# def draft_and_send_email():
"""
Expects JSON:
- product_desc: str
- recipient_name: str
- recipient_email: str
- sender_name: str
- subject (optional): str
"""
print("Drafting and sending email...")
# Initialize the EmailAgent
agent = EmailAgent()

# Change the below values as per your requirement
recipient_name = "BHarat Kamble"
recipient_email = "bharatekamble@gmail.com"
subject = "Exciting Product Offer Just For You!"
product_name = "real estate"
product_description = "A luxurious real estate property that offers stunning views and modern amenities."

# Validate the required fields
required_fields = [product_name, recipient_email]
if not all(required_fields):
    # return jsonify({"error": f"Missing fields. Required: {required_fields}"}), 400
    print(f"Missing fields. Required: {required_fields}")
    exit(1)

# Draft the email
try:
    email_body = agent.draft_marketing_email(product_name, product_description, recipient_name)
    print(f"Drafted email body: {email_body}")
except Exception as e:
    # return jsonify({"error": f"Could not generate mail: {str(e)}"}), 500
    print(f"Could not generate mail: {str(e)}")

# Send the email
try:
    agent.send_marketing_email(recipient_email, subject, email_body)
    print(f"Email sent to {recipient_email} successfully!")
except Exception as e:
    # return jsonify({"error": f"Could not send mail: {str(e)}"}), 500
    print(f"Could not send mail: {str(e)}")

# return jsonify({
#     "message": f"Marketing email sent to {recipient_email}",
#     "draft": email_body
# }), 200


# if __name__ == '__main__':
#     app.run(debug=True)

