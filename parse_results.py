import csv

f = open("results1.txt" ,"r")
lines = f.readlines()


with open("results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    i = 0
    while i < len(lines):
        if lines[i].strip() == "Results:":
            sims = lines[i+1].split()[1].strip()
            gridsize = lines[i+2].split()[3].strip()
            ranks = lines[i+3].split()[1].strip()
            strat = lines[i+4].split()[1].strip()
            time = lines[i+5].split()[5].strip()
            nodes = int(ranks)/6
            writer.writerow([nodes, ranks, sims, gridsize, 'unknown', strat, time])
            i+=6
        else:
            i += 1
        
    
