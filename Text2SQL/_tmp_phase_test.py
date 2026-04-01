from text_to_sql_agent import AgentTextToSql

queries = [
    "Trova un forno quantistico al plutonio 99 teglie invisibili con cliente zyxw9876",
    "Cerco forni combinati a gas 6 teglie con doppia camera e ultrafiltri", 
    "Mostrami 500 prodotti"
]

agent = AgentTextToSql()
for i, q in enumerate(queries, 1):
    print("\n" + "="*120)
    print(f"TEST {i}: {q}")
    print("="*120)
    result = agent.process_request_with_execution(q, max_retries=4)
    print(f"SUCCESS: {result.get('success')}")
    print(f"ATTEMPTS: {result.get('attempts')}")
    if result.get('success'):
        qr = result.get('query_results', {})
        print(f"ROW_COUNT: {qr.get('row_count')}")
        print(f"IS_BROAD_RESULT: {qr.get('is_broad_result', False)}")
        print(f"SQL: {result.get('sql_query')}")
    else:
        print(f"ERROR: {result.get('error')}")
        print(f"LAST_SQL: {result.get('last_sql_query')}")
