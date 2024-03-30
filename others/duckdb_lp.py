import duckdb

def get_logical_plan(sql_query):
    # Initialize a DuckDB connection
    conn = duckdb.connect(database=':memory:', read_only=False)

    # Create a table
    # CREATE TABLE integers(i INTEGER);
    conn.execute("CREATE TABLE integers(i INTEGER);")
    # INSERT INTO integers VALUES (1), (2), (3), (NULL);
    conn.execute("INSERT INTO integers VALUES (1), (2), (3), (NULL);")

    conn.execute("CREATE TABLE t1(a INTEGER, b INTEGER, p INTEGER, q INTEGER, r INTEGER);")
    conn.execute("INSERT INTO t1 VALUES (1, 2, 3, 4, 5), (6, 7, 8, 9, 10);")

    conn.execute("CREATE TABLE t2(c INTEGER, d INTEGER);")
    conn.execute("INSERT INTO t2 VALUES (1, 2), (3, 4);")

    conn.execute("CREATE TABLE t3(e INTEGER, f INTEGER);")
    conn.execute("INSERT INTO t3 VALUES (1, 2), (3, 4);")

    conn.execute("CREATE TABLE customer(id INTEGER, name VARCHAR);")
    conn.execute("INSERT INTO customer VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie');")

    conn.execute("CREATE TABLE sales(s_id VARCHAR, c_id INTEGER, val INTEGER);")
    conn.execute("INSERT INTO sales VALUES ('A', 1, 10), ('B', 1, 20), ('C', 2, 30), ('D', 2, 40);")

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