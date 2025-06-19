# Copyright (c) 2024, Frappe Technologies and contributors
# For license information, please see license.txt

# import json

# from sqlalchemy.sql.operators import from_

import frappe
from frappe.model.document import Document
from langchain_core.tools import Tool
from langchain.agents import AgentType, ZeroShotAgent, AgentExecutor
# from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
# from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAI
# from pydantic.v1 import BaseModel, Field
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
# from langchain_community.llms import VertexAI
import ast
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

class ChatbotAI(Document):
	# begin: auto-generated types
	# This code is auto-generated. Do not modify anything in this block.

	from typing import TYPE_CHECKING

	if TYPE_CHECKING:
		from frappe.types import DF

		response: DF.LongText | None
		user_input: DF.SmallText | None
	# end: auto-generated types
	pass

	# def __init__(self, *args, **kwargs	):
		# Replace 'your_api_key' with your actual Gemini API key
		# self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
		# 								  api_key="AIzaSyBWMxWV9Iw2QKGOaS5IZl0YZe23BsEJ4Ok")
		# self.llm = ''
		# class add(BaseModel):
		# 	"""Add two integers."""
		#
		# 	a: int = Field(..., description="First integer")
		# 	b: int = Field(..., description="Second integer")
		#
		# class multiply(BaseModel):
		# 	"""Multiply two integers."""
		#
		# 	a: int = Field(..., description="First integer")
		# 	b: int = Field(..., description="Second integer")

		# self.tools = [add, multiply]

		# self.llm_with_tools = self.llm.bind_tools(self.tools)

		# self.structured_output = self.llm.with_structured_output(multiply)

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


# @frappe.whitelist()
# def get_user_input(user_input):
# 	print(user_input)
# 	# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key="AIzaSyBWMxWV9Iw2QKGOaS5IZl0YZe23BsEJ4Ok")
#
# 	class Joke(BaseModel):
# 		"""Joke to tell user."""
#
# 		setup: str = Field(..., description="The setup of the joke")
# 		punchline: str = Field(..., description="The punchline of the joke")
#
# 	examples = [
# 		HumanMessage("Tell me a joke about planes", name="example_user"),
# 		AIMessage(
# 			"",
# 			name="example_assistant",
# 			tool_calls=[
# 				{
# 					"name": "joke",
# 					"args": {
# 						"setup": "Why don't planes ever get tired?",
# 						"punchline": "Because they have rest wings!",
# 						"rating": 2,
# 					},
# 					"id": "1",
# 				}
# 			],
# 		),
# 		# Most tool-calling models expect a ToolMessage(s) to follow an AIMessage with tool calls.
# 		ToolMessage("", tool_call_id="1"),
# 		# Some models also expect an AIMessage to follow any ToolMessages,
# 		# so you may need to add an AIMessage here.
# 		HumanMessage("Tell me another joke about planes", name="example_user"),
# 		AIMessage(
# 			"",
# 			name="example_assistant",
# 			tool_calls=[
# 				{
# 					"name": "joke",
# 					"args": {
# 						"setup": "Cargo",
# 						"punchline": "Cargo 'vroom vroom', but planes go 'zoom zoom'!",
# 						"rating": 10,
# 					},
# 					"id": "2",
# 				}
# 			],
# 		),
# 		ToolMessage("", tool_call_id="2"),
# 		HumanMessage("Now about caterpillars", name="example_user"),
# 		AIMessage(
# 			"",
# 			tool_calls=[
# 				{
# 					"name": "joke",
# 					"args": {
# 						"setup": "Caterpillar",
# 						"punchline": "Caterpillar really slow, but watch me turn into a butterfly and steal the show!",
# 						"rating": 5,
# 					},
# 					"id": "3",
# 				}
# 			],
# 		),
# 		ToolMessage("", tool_call_id="3"),
# 	]
#
# 	system = """You are a hilarious comedian. Your specialty is knock-knock jokes. \
# 	Return a joke which has the setup (the response to "Who's there?") \
# 	and the final punchline (the response to "<setup> who?")."""
# 	structured_llm = llm.with_structured_output(Joke)
#
# 	prompt = ChatPromptTemplate.from_messages(
# 		[("system", system), ("placeholder", "{examples}"), ("human", "{input}")]
# 	)
# 	few_shot_structured_llm = prompt | structured_llm
# 	response =few_shot_structured_llm.invoke({"input": user_input, "examples": examples})
# 	frappe.db.set_value("Chatbot AI", "CBAI0001", 'response', response.punchline)


@frappe.whitelist()
def get_response(user_input):

	llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key="AIzaSyBWMxWV9Iw2QKGOaS5IZl0YZe23BsEJ4Ok")
	# from google.cloud import aiplatform
	# aiplatform.init(project="gen-lang-client-0659427914", location="us-central1")
	# llm = VertexAI(model="gemini-1.5-flash-002", api_key="AIzaSyBWMxWV9Iw2QKGOaS5IZl0YZe23BsEJ4Ok")
	agent_kwargs = {
		"extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
		"system_message":
			SystemMessage(content="""
            You are a friendly ERP bot who analyzes data of items, customer and forecasting and provide results by performing the actions.
            Your Task is to execute the provided tools and provide the results by analyzing the response of tools. \

            Provide your introduction in short summary.\

            While answering to each user input you have to follow this instructions.
            -Step 1: Classify the user input in below provided 'task lists'. If you can't find any matching task from the lists, reply 'I can not answer this'.\
            -Step 2: Follow the instructions provided in each tasks and after executing the functions mentioned in the instruction to get data from system.\
            -Step 3: Analyse the data returned from the function and give reply in detail.\

            Below is your 'task list' and step you need to follow to accomplish each task.\

			1) Tell your name, Your name is `ERP Agent` and what can you do.

            2)Provide Product Stock Information : In this task you will provide the information about the stock available in warehouse. To perform this action you will need 1 detail from user which is <item_code>. You need to use `GetStockDetails` for this task.\

            3)Provide Product Price Information :In this task you will first execute the function `GetPriceDetails` and from the results provide the price detail. Remember don't use your knowledge to answer this question,the answer is strictly from function response.To perform this action you will need 1 detail from user which is <item_code>.\

            4)Provide Sales Analysis of customer : In this task you need to analyse the list of orders provided to you. To perform this task you need to use `CustomerSalesAnalysis`.To perform this action you will need <customer_name> from user.\

			5)Provide demand forecasting of items: in this task you need to analyse sales of items from sales invoices provided to you and apply ARIMA technique on the data.
			To perform this task you need to `ArimaForecastTool`. To perform this action you will need item code or name and dates.

			6) By analyzing and memorizing all the data, you can give detail answers for business related insights about products, sales, demand forecasting.

            Things to consider while replying :
            - Ensure you have all necessary information required for the selected function, and adhere strictly to the function's guidelines. Proper execution depends on this level of detail and accuracy.
            - System currency is AED so provide the details accordingly.
            - Outstanding details are different from sales analysis.
            """)

	}

	def get_stock_details(stock_inquiry_string):
		item = stock_inquiry_string.strip()
		sql_data = frappe.db.sql(
			"""SELECT it.item_code,it.item_name,bn.actual_qty,bn.warehouse
			FROM `tabBin` bn inner join `tabItem` it
			on it.name = bn.item_code
			WHERE it.item_name like %s and warehouse='Finished Goods - BD' """,
			('%' + item + '%'), as_dict=1)

		return f"""
	        Below are the details of stock from the system, quickly analyse this and don't look for too many options and provide the information as per user query.\n
	        {sql_data}
	        """

	def sales_analysis(customer_name):
		customer = customer_name.strip()
		sales_data = get_sales_data(customer)
		return f"""
	        Below are the details of sales orders for {customer}, analyse this and if sales_data is empty, just stop and give output please provide customer name, don't look for too many options and provide the information as per user query.\n
	        {sales_data}
	        """

	def get_forecast_data():
		# Load data from CSV file
		file_path = '/home/danyal/Downloads/expanded_demand_forecasting_data.csv'  # Adjust the path as needed
		sales_data = pd.read_csv(file_path)

		# Convert 'posting_date' to datetime and set as index
		sales_data['posting_date'] = pd.to_datetime(sales_data['posting_date'])
		sales_data = sales_data.set_index('posting_date')
		# //////////////////////////////////////////////

		item_groups = sales_data.groupby('item_name')

		# Create empty dictionaries to store item-wise forecasts
		item_forecasts = {}
		total_actual_amount = []
		total_actual_qty = []
		total_actual_rate = []
		total_forecast_rate = []
		total_forecast_amount = []
		total_forecast_qty = []

		# Loop over each item to forecast item-wise
		for item, item_data in item_groups:
			# Use 'amount' and 'qty' columns for each item
			item_data = item_data[['amount', 'qty', 'rate']]

			# Split data into training and testing sets
			split_index = int(len(item_data) * 0.9)
			train_data = item_data[:split_index]
			test_data = item_data[split_index:]

			# Fit ARIMA model for 'amount' (item-wise)
			model_amount = ARIMA(train_data['amount'], order=(2, 0, 3))
			model_fit_amount = model_amount.fit()

			# Generate forecasts for 'amount'
			forecasts_amount = model_fit_amount.predict(start=len(train_data),
														end=len(item_data) - 1)

			# Fit ARIMA model for 'qty' (item-wise)
			model_qty = ARIMA(train_data['qty'], order=(2, 0, 3))
			model_fit_qty = model_qty.fit()

			# Generate forecasts for 'qty'
			forecasts_qty = model_fit_qty.predict(start=len(train_data), end=len(item_data) - 1)

			# Fit ARIMA model for 'rate' (item-wise)
			model_rate = ARIMA(train_data['rate'], order=(2, 0, 3))
			model_fit_rate = model_rate.fit()

			# Generate forecasts for 'rate'
			forecasts_rate = model_fit_rate.predict(start=len(train_data), end=len(item_data) - 1)

			# Store item-wise forecasts
			item_forecasts[item] = {
				'Actual Amount': test_data['amount'],
				'Forecasted Amount': forecasts_amount,
				'Actual Qty': test_data['qty'],
				'Forecasted Qty': forecasts_qty,
				'Actual Rate': test_data['rate'],
				'Forecasted Rate': forecasts_rate
			}

			# Aggregate totals
			total_actual_amount.append(test_data['amount'].sum())
			total_forecast_amount.append(forecasts_amount.sum())
			total_actual_qty.append(test_data['qty'].sum())
			total_forecast_qty.append(forecasts_qty.sum())
			total_actual_rate.append(test_data['rate'].mean())
			total_forecast_rate.append(forecasts_rate.mean())

		# Calculate total sales (amount and qty)
		total_actual_amount = sum(total_actual_amount)
		total_forecast_amount = sum(total_forecast_amount)
		total_actual_qty = sum(total_actual_qty)
		total_forecast_qty = sum(total_forecast_qty)
		total_actual_rate = sum(total_actual_rate) / len(total_actual_rate)
		total_forecast_rate = sum(total_forecast_rate) / len(total_forecast_rate)

		# Return both item-wise and total forecasts
		return {
			'Item-Wise Forecasts': item_forecasts,
			'Total Actual Amount': total_actual_amount,
			'Total Forecasted Amount': total_forecast_amount,
			'Total Actual Qty': total_actual_qty,
			'Total Forecasted Qty': total_forecast_qty,
			'Total Actual Rate': total_actual_rate,
			'Total Forecasted Rate': total_forecast_rate
		}
		# ////////////////////////////////////////////////////////////////////////
		# Use only the 'amount' column for the model
		# sales_data = sales_data[['amount', 'qty']]
		#
		# # Split data into training and testing sets
		# split_index = int(len(sales_data) * 0.9)
		# train_data = sales_data[:split_index]
		# test_data = sales_data[split_index:]
		#
		# # Fit ARIMA model
		# model = ARIMA(train_data['amount'], order=(2, 0, 3))
		# model_fit = model.fit()
		#
		# # Generate forecasts
		# forecasts_amount = model_fit.predict(start=len(train_data), end=len(sales_data) - 1)
		#
		# model = ARIMA(train_data['qty'], order=(2, 0, 3))
		# model_fit = model.fit()
		#
		# # Generate forecasts
		# forecasts_qty = model_fit.predict(start=len(train_data), end=len(sales_data) - 1)
		#
		# # Evaluate model performance for 'amount'
		# rmse = mean_squared_error(test_data['amount'], forecasts_amount, squared=False)
		# mean_actual = test_data['amount'].mean()
		# rmse_ratio = rmse / mean_actual
		# print(f'RMSE as a percentage of the mean actual: {rmse_ratio * 100}%')
		#
		# # Evaluate model performance for 'qty'
		# rmse_qty = mean_squared_error(test_data['qty'], forecasts_qty, squared=False)
		# mean_actual_qty = test_data['qty'].mean()
		# rmse_ratio_qty = rmse_qty / mean_actual_qty
		# print(f'RMSE for qty as a percentage of the mean actual: {rmse_ratio_qty * 100}%')
		#
		# # Return both forecasts
		# return {
		# 	'Actual Amount': test_data['amount'],
		# 	'Forecasted Amount': forecasts_amount,
		# 	'Actual Qty': test_data['qty'],
		# 	'Forecasted Qty': forecasts_qty
		# }


	def get_sales_inv(dates):
		# dates1 = ast.literal_eval(dates)
		# f_date = dates1.get('from_date')
		# t_date = dates1.get('to_date')
		# sales_inv_detail = frappe.db.sql("""
		# 	    select si.customer_name,si.posting_date,sii.item_code,sii.item_name,sii.qty,sii.rate,sii.amount
		# 	    from `tabSales Invoice` si
		# 	    inner join `tabSales Invoice Item` sii on si.name = sii.parent
		# 	    where si.docstatus = 1 and si.posting_date between '{0}' and '{1}'""".format(f_date, t_date),as_dict=1)

		forecasted_result = get_forecast_data()
		# Generate random sales data
		# np.random.seed(42)
		# print(sales_inv_detail)
		# sales_data = pd.DataFrame({'Sales': sales_inv_detail})
		# # sales_data['Date'] = pd.date_range(start='2023-01-01', periods=50)
		# sales_data = sales_data.set_index('Date')
		#
		# # Split data into training and testing sets
		# train_data = sales_data[:-10]
		# test_data = sales_data[-10:]
		#
		# # Fit ARIMA model
		# model = ARIMA(train_data['Sales'], order=(5, 1, 0))  # Example order, adjust as needed
		# model_fit = model.fit()
		#
		# # Generate forecasts
		# forecasts = model_fit.predict(start=len(train_data), end=len(sales_data) - 1)
		#
		# # Evaluate model performance
		# rmse = mean_squared_error(test_data['Sales'], forecasts, squared=False)
		#
		# # Create table for presentation
		# results_table = pd.DataFrame({'Actual': test_data['Sales'], 'Forecasted': forecasts})
		# results_table['Date'] = test_data.index
		# results_table = results_table[['Date', 'Actual', 'Forecasted']]
		# results_table = results_table.append({'Date': 'RMSE', 'Actual': None, 'Forecasted': rmse},
		# 									 ignore_index=True)
		#
		# print(results_table)

		return f"""
			        Below are the details of {forecasted_result}, analyse the forecasted result and provide a suitable answer.
			        """

	def create_po(data):
		import json
		data_str = "{0}".format(data)
		# Replace single quotes with double quotes for valid JSON format
		data_str = data_str.replace("'", '"')
		data = json.loads(data_str)

		po_doc = frappe.new_doc("Purchase Order")
		po_doc.supplier = data.get('supplier')
		po_doc.company = 'BLOX(Demo)'
		po_doc.transaction_date = data.get('delivery_date')
		po_doc.schedule_date = data.get('delivery_date')
		po_doc.currency= 'AED'
		item = frappe.get_value("Item", {'item_name': data.get('item')}, ['item_code', 'item_name'], as_dict=1)
		po_doc.append("items", {
			'item_code': item.item_code,
			'item_name': item.item_name,
			'rate': data.get('price'),
			'qty': data.get('forecasted_qty'),
			'warehouse': 'Stores - BD'
		})
		po_doc.save()
		return f""" Purchase Order has been created for item {data.get('item')}"""


	stock_detail_tool = Tool(
		name="GetStockDetails",
		func=get_stock_details,
		description = """
                Description: The 'GetStockDetails' function to get details of stock from inventory. Here's what you
                need:

                1. A SINGLE STRING in the format: ```Item Code```.

                An example function input for an order might look like: ```Apple```
                It is important to remember that the input should be formatted as a single string, not a list or multiple strings.

                A word of caution: if any information is unclear, incomplete, or not confirmed, the function might
                not work correctly.

                """
	)

	customer_sales_analysis_tool = Tool(
		name="CustomerSalesAnalysis",
		func=sales_analysis,
		description = """
                Description: The 'CustomerSalesAnalysis' function to get details of list of sales orders for a customer. Here's what you
                need:

                1. Ask the user for A SINGLE STRING in the format: ```Customer Name```.
				2. If you don't have customer's name just reply "Please provide customer name"
                An example function input for an order might look like: ```Danyal khan```
                It is important to remember that the input should be formatted as a single string, not a list or multiple strings.

                A word of caution: if any information is unclear, incomplete, or not confirmed, the function might
                not work correctly.

                """
	)


	arima_forecast_tool = Tool(
		name="ArimaForecastTool",
		func=get_sales_inv,
		description="""
	                Description: The 'ArimaForecastTool' function to get details of list of items from sales invoices. Here's what you
	                need:

					1. Run this function one time and save all the information for further insights answers.

					2. get_sales_inv function takes a dictionary as argument called dates.

	                3. Ask the user for dates in the format: ```From date till To date```.

					4. Analyse the actual and forecasted data from get_sales_inv and give insight answers related to product demand.

					An example function input for an order might look like: ``` '2024-09-01' and '2024-09-31' ```
	                It is important to remember that the input should be formatted as dictionary named as dates with keys as from_date and to_date,
	                values can be extracted from dictionary by using dates.get('from_date') otherwise function will not work.

	                A word of caution: if any information is unclear, incomplete, or not confirmed, the function might
	                not work correctly.

	                """
	)

	create_po = Tool(
		name="PurcahaseOrder",
		func=create_po,
		description="""
		                Description: The 'PurcahaseOrder' function is to create purchase order. Here's what you
		                need:

						1. create_po function takes a dictionary as argument called data, with keys like item, supplier, price, delivery_date, forecasted_qty.


						An example function dictionary of argument for an purchase order might look like: ```Sneakers, MA Inc., 222, 2024-10-10, 44  ```
		                It is important to remember that the input should be formatted as dictionary named as 'data' with keys as item, supplier, price, delivery_date and forecasted_qty
		                 otherwise function will not work.

		                A word of caution: if any information is unclear, incomplete, or not confirmed, the function might
		                not work correctly.

		                """
	)

	# Step 3: Pass the tools in ToolMessage format
	tools = [customer_sales_analysis_tool, stock_detail_tool, arima_forecast_tool, create_po]
	# llm_with_tools = llm.bind_tools(tools)

	# Simulating a user message and system tool message workflow
	# human_message = HumanMessage(content="3,5")
	# response = llm_with_tools.invoke("what is 8 + 10")

	agent = ZeroShotAgent.from_llm_and_tools(llm=llm, tools=tools)
	agent_executor = AgentExecutor.from_agent_and_tools(kwargs = agent_kwargs, agent=agent, tools=tools, verbose=True, max_iterations=15)
	response = agent_executor.invoke(user_input)
	print(f"LLM Response: {response}")
	# print(f"LLM Response: {response.action_input}")
	return response

	# frappe.db.set_value("Chatbot AI", "CBAI0001", 'response', response.get('output'))

	# Pass numbers as a single string separated by a comma
	# tool_message = ToolMessage(content="", tool_call_id='', tool_name="Addition Tool", tool_input={"numbers": "3,5"})

	# Example invocation
	# result = add_tool.run({"numbers": "3,5"})  # Calling the tool's run method with a single argument
	# print(f"Result: {result}")
