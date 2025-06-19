# Copyright (c) 2024, Frappe Technologies and contributors
# For license information, please see license.txt

# import frappe
from frappe.model.document import Document


class DemandForecasting(Document):
	# begin: auto-generated types
	# This code is auto-generated. Do not modify anything in this block.

	from typing import TYPE_CHECKING

	if TYPE_CHECKING:
		from frappe.types import DF

		amount: DF.Float
		customer_name: DF.Data | None
		item_code: DF.Data | None
		item_name: DF.Data | None
		posting_date: DF.Data | None
		qty: DF.Float
		rate: DF.Data | None
	# end: auto-generated types
	pass
