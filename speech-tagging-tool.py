import sys

if(len(sys.argv) < 2):
    print("Error!")
    print("No parameter passed!")
    print("Type -h option to see man.")
    exit
else:
    if(len(sys.argv) >= 2 and len(sys.argv) <= 4):
        if(sys.argv[1] == '-h'):
            print("{} man".format(sys.argv[0]))
            print("{} <option> <input_file> <output_file>\n".format(sys.argv[0]))
            print("<option>\t-\tdefines whether to return phonemes (-P) or words (-W) tagging")
            print("<input_file>\t-\tpath to sound file")
            print("<output_file>\t-\tpath to output file")
            print("-h\t\t-\thelp")
            exit
        else:
            print(sys.argv[1])
    else:
        print("Error!")
        print("Too many arguments passed!")
        print("Type -h option to see man.")
        exit
