from tempfile import mkstemp
from shutil import move
from os import remove, close

def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with open(abs_path,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    close(fh)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)

def change_delimiter(input_file_path, output_file_path, from_delimiter, to_delimiter):
    with open(input_file_path) as infile:
        with open(output_file_path, 'w') as outfile:
            for line in infile:
              fields = line.split(from_delimiter)
              outfile.write(to_delimiter.join(fields))
			  
			  
def remove_folder_contents(folder):
	for the_file in os.listdir(folder):
	    file_path = os.path.join(folder, the_file)
	    try:
	        if os.path.isfile(file_path):
	            os.unlink(file_path)
	        elif os.path.isdir(file_path): shutil.rmtree(file_path)
	    except Exception as e:
	        print(e)



def singleFile_to_multiFile(original_file, first_file_end_point, write_from, write_till, folder_name):
	
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)

	with open(original_file) as input_file:
		line_count = 0
		sample_count = 0
		for line in input_file:
			line_count = line_count + 1
			if (line_count == 1):
				sample_count = sample_count + 1
				current_file_name = folder_name+"\\"+str(sample_count)+".csv"
				current_file  = open(current_file_name, 'w+')
			if (line_count == write_from):
				current_file.write(line.rstrip(' \n')+"\n")
			if (line_count > write_from and line_count <=write_till):
				current_file.write(line.rstrip(',\n')+"\n")
			if (line_count == first_file_end_point):
				line_count = 0