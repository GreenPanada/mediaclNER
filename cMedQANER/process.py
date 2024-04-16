with open("cMedQANER\\train.txt",'r',encoding='utf-8') as f:
    lines = f.readlines()

    tags = set()

    for line in lines:
        tags.add(line.rstrip().split(' ')[1])
    print(tags)
        