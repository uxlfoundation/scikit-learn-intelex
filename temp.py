if True:
    with open('requirements-doc.txt', 'r') as infile, open('abc.txt', 'w') as outfile:
        for line in infile:
            line = line.strip()
            if line and not line.startswith('#'):
                package = line.split('==')[0].strip()
                outfile.write(f"{package}\n")
