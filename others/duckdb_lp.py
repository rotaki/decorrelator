import duckdb

def get_logical_plan(sql_query):
    # Initialize a DuckDB connection
    conn = duckdb.connect(database=':memory:', read_only=False)

    # Create a table
    # CREATE TABLE integers(i INTEGER);
    conn.execute("CREATE TABLE integers(i INTEGER);")
    # INSERT INTO integers VALUES (1), (2), (3), (NULL);
    conn.execute("INSERT INTO integers VALUES (1), (2), (3), (NULL);")

    try:
        # 'all'
        # 'optimized_only'
        # 'physical_only'
        conn.execute(f"SET explain_output = 'all';")

        plan = conn.execute(f"EXPLAIN {sql_query}").fetchall()
        
        for row in plan:
            print(row[1])

        result = conn.execute(sql_query).fetchall()
        print("Number of rows: ", len(result))
        for row in result:
            print(row) 

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the connection
        conn.close()

if __name__ == "__main__":
    # Read SQL from input
    sql_query = input("Enter the SQL query: ")
    
    # Get the logical plan
    get_logical_plan(sql_query)