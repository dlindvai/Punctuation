import sys

with open(sys.argv[2], 'a') as output_file, open(sys.argv[1], 'r') as input_file:
	for line in input_file:
		for word in line.split():
			if (word == '.PER'):
				word = word.replace(word, 'P')
				output_file.write(word + ' ')
			elif (word == ',COM'):
				word = word.replace(word, 'C')
				output_file.write(word + ' ')
			else:
				word = word.replace(word, 'W')
				output_file.write(word + ' ')
