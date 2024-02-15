
import sys

class CorpusAlgorithm:

	def __init__(self, log_fd=sys.stderr):
		self.log_fd = log_fd

	def logStep(self, text="corpus algorithm"):
		print("*", text, file=self.log_fd)

	def processCorpus(self):
		pass

