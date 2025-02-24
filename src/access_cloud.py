
class Access:
	def bearer_is_valid(self, token: str) -> bool:
		return len(token) == 20
