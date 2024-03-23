use datafusion::arrow::array::{ArrayRef, Int32Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::prelude::SessionContext;
use std::io::{self, Write};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let mem_table = create_memtable()?;

    // create local execution context
    let ctx = SessionContext::new();

    // Register the in-memory table containing the data
    ctx.register_table("integers", Arc::new(mem_table))?;

    // Prompt the user for a SQL query
    print!("Enter the SQL query: ");
    io::stdout().flush().unwrap(); // Ensure the prompt is displayed immediately
    let mut sql_query = String::new();
    io::stdin().read_line(&mut sql_query).unwrap();
    let sql_query = sql_query.trim(); // Trim newline and leading/trailing whitespace

    // Create the logical plan
    let logical_plan = ctx.sql(sql_query).await?;
    let logical_plan = logical_plan.into_unoptimized_plan();

    // Print the logical plan
    println!("Logical Plan:\n{:?}", logical_plan);

    // Optimize the logical plan
    let optimized_logical_plan = ctx.optimize(&logical_plan)?;

    // Print the optimized logical plan
    println!("\nOptimized Logical Plan:\n{:?}", optimized_logical_plan);

    // Execute the query
    let df = ctx.sql(sql_query).await?;

    // Collect the results
    let results = df.collect().await?;

    // Pretty print the result
    println!("{:?}", results);

    Ok(())
}

fn create_memtable() -> Result<MemTable> {
    MemTable::try_new(get_schema(), vec![vec![create_record_batch()?]])
}

fn create_record_batch() -> Result<RecordBatch> {
    Ok(RecordBatch::try_new(
        get_schema(),
        vec![Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(3), None])) as ArrayRef],
    )
    .unwrap())
}

fn get_schema() -> SchemaRef {
    let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, true)]));

    schema
}
