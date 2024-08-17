with open('/home/mengyuan/workDir/seed_data5-16/tree_V4') as f:
    father = set()
    child = set()
    for line in f:
        lines = line.strip().split(',IsA,')
        father.add(lines[-1])
        child.add(lines[0])
    print(father-child)
