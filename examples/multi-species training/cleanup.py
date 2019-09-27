import os
import json

# Clean up the input files for a fresh run.

if __name__ == "__main__" :
    filesfp = open("sets.json", "r")
    files = json.load(filesfp)
    filesfp.close()

    fp = open("ml.json", "r+")
    data = json.load(fp)

    fp.close()
    os.remove("ml.json")
    fp = open("ml.json", "w")

    data["data"]["hypers"] = False
    data["data"]["a"] = False
    data["data"]["s"] = False
    data["data"]["l"] = False
    data["data"]["trainers"] = False
    json.dump(data, fp, indent = "\t")
    fp.close()
    
    fp = open("ml_individ.json", "r+")
    data = json.load(fp)

    fp.close()
    os.remove("ml_individ.json")
    fp = open("ml_individ.json", "w")

    data["data"]["hypers"] = False
    data["data"]["a"] = False
    data["data"]["s"] = False
    data["data"]["l"] = False
    data["data"]["trainers"] = False
    json.dump(data, fp, indent = "\t")
    fp.close()

    for f in files["training"] :
        fp = open(f, "r+")
        data = json.load(fp)

        fp.close()
        os.remove(f)
        fp = open(f, "w")

        data["generated"] = False;
        data["complete"] = False;
        json.dump(data, fp, indent = "\t")
        fp.close()

        fp = open("ml_individ_" + f, "r+")
        data = json.load(fp)

        fp.close()
        os.remove("ml_individ_" + f)
        fp = open("ml_individ_" + f, "w")
        
        data["data"]["hypers"] = False
        data["data"]["a"] = False
        data["data"]["s"] = False
        data["data"]["l"] = False
        data["data"]["trainers"] = False
        json.dump(data, fp, indent = "\t")
        fp.close()
    for f in files["validation"] :
        fp = open(f, "r+")
        data = json.load(fp);

        fp.close()
        os.remove(f)
        fp = open(f, "w")
        
        data["generated"] = False
        data["complete"] = False
        json.dump(data, fp, indent = "\t")
        fp.close()
