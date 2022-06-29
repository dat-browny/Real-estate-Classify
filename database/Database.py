import logging
import pandas as pd
from tqdm import tqdm
from predict.Predict import predict_label
from neo4j import GraphDatabase, basic_auth

HOST = '103.252.1.131'
PORT = '7687'
AUTH_USERNAME = 'neo4j'
AUTH_PASSWORD = 'datalake'

class normalize_data:
    driver = GraphDatabase.driver('bolt://' + HOST + ':' + PORT, auth=basic_auth(AUTH_USERNAME, AUTH_PASSWORD),
                                  max_connection_lifetime=15 * 60, max_connection_pool_size=500)

    def __init__(self):
        # connect to neo4j
        self.db_neo4j = self.get_db()

    def close(self):
        self.driver.close()

    def get_db(self):
        return self.driver.session()
        
    def run_query(self, session, query):
        result = session.run(query)
        return result

    def get_content(self):
        session = self.db_neo4j
        num_records = session.run(
            "MATCH (n:Status) WHERE n.PostType='non Real Estate' RETURN count(n) as num_records").single()["num_records"]
        results =  session.run("MATCH (n:Status) WHERE n.PostType='non Real Estate' RETURN n.content, n.fb_id")
        contents = []
        fb_ids = []
        logging.info("Getting contents...")
        for record in tqdm(results, total=num_records):
            content = record["n.content"]
            if content is not None:
                contents.append(content)
                fb_ids.append(record["n.fb_id"])
        logging.info("Completed!")
        return contents, fb_ids
    
    def predict_label(self, output_dir):
        contents, fb_ids = self.get_content()
        df = pd.DataFrame(contents, columns=["contents"])
        df["fb_ids"] = fb_ids
        label = []
        logging.info("Predicting label...")
        for content in tqdm(df["contents"]):
            label.append(predict_label(content))
        logging.info("Complete predicting label!")
        logging.info("==========================")
        logging.info("Saving dataframe to path...")
        df['label'] = label
        df.to_pkl(output_dir)
        logging.info(f"Dataframe saved at '{output_dir}'")
        return df

    def update__post(self, dataframe):
        with self.driver.session() as session:
            logging.info("Updating topic type to Neo4j database...")
            for item in tqdm(dataframe):
                query_update = "match(n:Status{fb_id: \"" + str(item['fb_ids']) + "\"}) " + f"set n.Topic =  {item['label']}"
                self.run_query(session, query_update)
            logging.info("Update database sucessfully!")

