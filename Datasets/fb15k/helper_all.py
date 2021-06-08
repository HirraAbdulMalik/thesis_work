neg_samples = []
sibling_relationship = "/people/person/sibling_s./people/sibling_relationship/sibling"
spouse_relationship = "/people/person/spouse_s./people/marriage/spouse"
friendship_relationship = "/base/popstra/celebrity/friendship./base/popstra/friendship/participant"
parent_relation = "/people/person/parents"
child_relation = "/people/person/children"
all_lines = []
with open("original/test.txt", "r") as read_file:
	for line in read_file:		
		all_lines.append(line)
		#sep_line = line.split()
		#if sep_line[1] == spouse_relationship:
		#	neg_samples.append(sep_line[0] + "\t" + sibling_relationship + "\t" + sep_line[2] + "\n")	
		#if sep_line[1] == friendship_relationship:
		#	neg_samples.append(sep_line[0] + "\t" + sibling_relationship + "\t" + sep_line[2] + "\n")
		#if sep_line[1] == sibling_relationship:
		#	neg_samples.append(sep_line[0] + "\t" + friendship_relationship + "\t" + sep_line[2] + "\n")
#c = 0
for line in all_lines:
	sep_line = line.split()
	if sep_line[1] == spouse_relationship:
		all_entities = [line for line in all_lines if (sep_line[0] or sep_line[2]) in line] 
		a = [line for line in all_entities if (parent_relation or child_relation) in line]
		#print(a)
		for lin in a:
			print(lin+"\n")

#with open("Symmetry/People/negative_test_samples.txt", "w") as d_file: 
#	for line in neg_samples:
#	_file.write(line)
