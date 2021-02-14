neg_samples = []
sibling_relationship = "/people/person/sibling_s./people/sibling_relationship/sibling"
spouse_relationship = "/people/person/spouse_s./people/marriage/spouse"
friendship_relationship = "/base/popstra/celebrity/friendship./base/popstra/friendship/participant"
with open("Symmetry/People/test.txt", "r") as read_file:
	for line in read_file:		
		sep_line = line.split()
		if sep_line[1] == spouse_relationship:
			neg_samples.append(sep_line[0] + "\t" + sibling_relationship + "\t" + sep_line[2] + "\n")	
		if sep_line[1] == friendship_relationship:
			neg_samples.append(sep_line[0] + "\t" + sibling_relationship + "\t" + sep_line[2] + "\n")
		if sep_line[1] == sibling_relationship:
			neg_samples.append(sep_line[0] + "\t" + friendship_relationship + "\t" + sep_line[2] + "\n")


with open("Symmetry/People/negative_test_samples.txt", "w") as d_file: 
	for line in neg_samples:
		d_file.write(line)
