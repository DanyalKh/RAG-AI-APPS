# # Copyright (c) 2015, Frappe Technologies Pvt. Ltd. and Contributors
# # License: GNU General Public License v3. See license.txt
#
# # For license information, please see license.txt


import frappe
from frappe.model.document import Document
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field


class GenerativeAI(Document):
	pass
	def __init__(self):
		# Replace 'your_api_key' with your actual Gemini API key
		self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key="AIzaSyBWMxWV9Iw2QKGOaS5IZl0YZe23BsEJ4Ok")

		class add(BaseModel):
			"""Add two integers."""

			a: int = Field(..., description="First integer")
			b: int = Field(..., description="Second integer")

		class multiply(BaseModel):
			"""Multiply two integers."""

			a: int = Field(..., description="First integer")
			b: int = Field(..., description="Second integer")

		self.tools = [add, multiply]

		self.llm_with_tools = self.llm.bind_tools(self.tools)

		self.structured_output = self.llm.with_structured_output(multiply)

		# response= s.invoke(query)
		# # Prompt the LLM with a question
		# prompt = "What is the capital of France?"

		# # Create a HumanMessage object
		# messages = [HumanMessage(content=prompt)]

		# # Get the response
		# response = llm(messages)

		# # Print the response
		# print(response.content)


	def get_stock_details(self, stock_inquiry_string):
		item = stock_inquiry_string.strip()
		sql_data = frappe.db.sql(
			"""SELECT it.item_code,it.item_name,bn.actual_qty,bn.warehouse FROM `tabBin` bn inner join `tabItem` it on it.name = bn.item_code WHERE it.item_name like %s""",
			('%' + item + '%'), as_dict=1)

		return f"""
	        Below are the details of stock from the system, analyse this and provide the information as per user query.\n
	        {sql_data}
	        """

	def get_price_details(self, price_inquiry_string):
		item = price_inquiry_string.strip()
		sql_data = frappe.db.sql(
			"""SELECT ip.item_code,ip.item_name,ip.price_list_rate FROM `tabItem Price` ip WHERE ip.item_name like %s and ip.selling = 1""",
			('%' + item + '%'), as_dict=1)

		return f"""
	        Below are the details of price from the system, analyse this and provide the information as per user query.\n
	        {sql_data}
	        """

	def sales_analysis(self, customer_name):
		customer = customer_name.strip()
		sales_data = get_sales_data(customer)
		return f"""
	        Below are the details of sales orders for {customer}, analyse this and provide the information as per user query.\n
	        {sales_data}
	        """

	def get_products(self, dummy_string):
		print(dummy_string)
		items = get_item_details()
		return f"""
	            Below are the list of items, analyse this and provide the information as per user query.\n
	            {items}
	            """

	def get_outstanding_invoices(self, customer_name):
		customer = customer_name.strip()
		invoices = get_outstanding_amount(customer)
		return f"""
	            Below are the list of pending invoices for customer {customer}, analyse this and provide the information as per user query.\n
	            {invoices}
	            """

	def get_customer_credit(self, customer_string):
		customer = customer_string.strip()
		credit_amount = get_credit_limit_from_db(customer)
		return f"""
	            Below is the credit amount provided to {customer}, analyse this and provide the information as per user query.\n
	            {credit_amount}
	            """


def get_sales_data(customer_name):
	sales_detail = frappe.db.sql("""
	    select so.customer_name,so.transaction_date,soi.item_name,soi.qty,soi.rate,soi.amount from `tabSales Order` so inner join `tabSales Order Item` soi on so.name = soi.parent where so.docstatus = 1 and so.customer like %s

	    """, ('%' + customer_name + '%'), as_dict=1)

	return sales_detail


def get_item_details():
	item_detail = frappe.db.sql("""
	    select item_code,item_name,brand,item_group from `tabItem` where disabled = 0
	    """, as_dict=1)
	return item_detail


def get_outstanding_amount(customer):
	outstanding_invoices = frappe.db.sql("""
	    select si.customer,si.posting_date,si.due_date,si.outstanding_amount from `tabSales Invoice` si where si.outstanding_amount > 0 and si.customer like %s
	    """, ('%' + customer + '%'), as_dict=1)
	return outstanding_invoices


def get_credit_limit_from_db(customer):
	credit_limit = frappe.db.sql("""
	    select ccl.credit_limit from `tabCustomer Credit Limit` ccl where ccl.parent like %s
	    """, ('%' + customer + '%'), as_dict=1)

	return credit_limit[0]['credit_limit']


@frappe.whitelist()
def get_user_input(user_input):
	bot = GenerativeAI()
	if not user_input:
		return 'No input provided'
	return bot.structured_output.invoke(user_input)


