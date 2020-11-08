import re
import sys

with open(sys.argv[2], 'a') as output_file, open(sys.argv[1], 'r') as input_file:
	for line in input_file:
		line = line.replace('\n', '')
		line = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", line)
		line = line.lower()
		line = line.replace('!', '.').replace(':', ',').replace(';', '.').replace('-', ',').replace('?', '.')
		line = line.replace('.', '.PER').replace(',', ',COM')
		output_file.write(line)
