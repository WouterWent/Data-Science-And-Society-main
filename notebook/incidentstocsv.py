import psycopg2
import configparser
import csv

# Load config
config = configparser.ConfigParser()
config.read('Config.txt')

# Get database configurations
db_config = config['DATABASE']

# Connect to db
try:
    cnx = psycopg2.connect(
        user=db_config['USER'],
        password=db_config['PASSWORD'],
        host=db_config['HOST'],
        port=db_config['PORT'],
        database=db_config['NAME']
    )
    
    print("Connected successfully!")
    cur = cnx.cursor()
####Query###########################################
    cur.execute("""
SELECT * FROM incidentdata;
    """)

    #Fetch tbl names
    table_names = cur.fetchall()
#    for table in table_names:
#        print(table)


    
# Open a CSV file for writing
    with open('incident.csv', 'w', newline='') as csvfile:
        # Write the header row
        writer = csv.writer(csvfile)
        writer.writerow([col[0] for col in cur.description])

        # Iterate over the rows in the table and write each row to the CSV file
        for row in table_names:
            writer.writerow(row)

    cnx.close()

except Exception as e:
    print(f"Error: {e}")
