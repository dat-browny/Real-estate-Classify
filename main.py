from database.Database import normalize_data 

if __name__ == "__main__":
    neo4j = normalize_data()
    df = neo4j.predict_label(path)
    neo4j.update__post()

