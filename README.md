# TODO
* [] Add SQL parser that parses the query into a tree
  * [x] SELECT
  * [x] FROM
  * [x] WHERE
  * [x] JOIN
  * [] ORDER BY
  * [] LIMIT
  * [x] Aggregate/Group By/Having
  * [] Subquery
    * [] Subquery in SELECT
    * [] Subquery in FROM
    * [] Subquery in WHERE
    * [] Others...
* [] Add a column_id manager that assigns a unique id to each column
  * [x] Query without subquery
  * [] Subquery
* [x] Show logical plan of other databases
  * [x] DuckDB
  * [x] Datafusion