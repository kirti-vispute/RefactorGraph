# Error Analysis Report (Phase 11 -- validation split only)

Model: Phase 10 tuned hybrid (models/hybrid_tuned/, heads=4 hidden_dim=128 dropout=0.103). Predictions re-derived per VAL source file by re-parsing with ml.preprocessing.ast_parser in the same node order ml/graph/graph_builder.py uses, so every prediction is attributed back to a concrete repo/file/qualname/lineno. TEST split is never loaded by this script.

False positive (FP) = predicted smelly, label says not smelly. False negative (FN) = predicted not smelly, label says smelly. "Hardest" FPs = highest-confidence wrong positives (prob closest to 1.0 while label=0). "Hardest" FNs = most confidently missed positives (prob closest to 0.0 while label=1) -- these are true smells the model was most wrong about, not just borderline 0.5 cases.

## long_method

- n=14802 positives=1162 precision=0.812 recall=0.851 f1=0.831 (tp=989 fp=229 fn=173 tn=13411)

### Hardest false positives (top 15 by confidence)

| repo | file | qualname | line | prob | metrics |
|---|---|---|---|---|---|
| sqlalchemy | lib/sqlalchemy/orm/_orm_constructors.py | relationship | 1059 | 1.000 | statement_count=2 |
| sqlalchemy | lib/sqlalchemy/sql/selectable.py | HasCTE.cte | 2714 | 1.000 | statement_count=2 |
| sphinx | sphinx/builders/linkcheck.py | CheckExternalLinksBuilder.process_result | 112 | 1.000 | statement_count=7 |
| sqlalchemy | lib/sqlalchemy/orm/_orm_constructors.py | mapped_column | 99 | 1.000 | statement_count=2 |
| sqlalchemy | lib/sqlalchemy/engine/base.py | Connection.execution_options | 261 | 1.000 | statement_count=6 |
| sqlalchemy | lib/sqlalchemy/dialects/postgresql/base.py | PGDialect._columns_query | 4294 | 1.000 | statement_count=12 |
| sqlalchemy | lib/sqlalchemy/dialects/mssql/base.py | MSSQLCompiler._render_json_extract_from_binary | 2489 | 1.000 | statement_count=13 |
| sqlalchemy | lib/sqlalchemy/engine/cursor.py | CursorResultMetaData._merge_cursor_description | 543 | 1.000 | statement_count=14 |
| sqlalchemy | lib/sqlalchemy/dialects/postgresql/base.py | PGDialect._index_query | 5170 | 1.000 | statement_count=10 |
| sqlalchemy | lib/sqlalchemy/orm/relationships.py | _JoinCondition._determine_joins | 2466 | 1.000 | statement_count=14 |
| sphinx | sphinx/environment/collectors/toctree.py | TocTreeCollector.process_doc | 64 | 1.000 | statement_count=9 |
| sphinx | sphinx/ext/todo.py | latex_visit_todo_node | 205 | 0.999 | statement_count=11 |
| sqlalchemy | lib/sqlalchemy/sql/ddl.py | SchemaDropper.visit_metadata | 1576 | 0.999 | statement_count=14 |
| sqlalchemy | lib/sqlalchemy/sql/coercions.py | expect_col_expression_collection | 446 | 0.999 | statement_count=14 |
| sphinx | sphinx/ext/autodoc/_dynamic/_docstrings.py | _attr_docs_for_props | 55 | 0.998 | statement_count=10 |

### Hardest false negatives (top 15 by confidence)

| repo | file | qualname | line | prob | metrics |
|---|---|---|---|---|---|
| sqlalchemy | lib/sqlalchemy/orm/collections.py | __go | 1543 | 0.000 | statement_count=15 |
| sphinx | sphinx/ext/apidoc/_cli.py | _parse_args | 282 | 0.000 | statement_count=17 |
| sphinx | sphinx/cmd/build.py | build_main | 395 | 0.000 | statement_count=18 |
| sphinx | utils/generate_snowball.py | regenerate_javascript | 99 | 0.001 | statement_count=18 |
| sqlalchemy | lib/sqlalchemy/sql/compiler.py | SQLCompiler.visit_unary | 3264 | 0.001 | statement_count=19 |
| sqlalchemy | lib/sqlalchemy/orm/mapper.py | Mapper._sorted_tables | 4014 | 0.002 | statement_count=15 |
| sphinx | sphinx/util/inspect.py | isattributedescriptor | 357 | 0.002 | statement_count=15 |
| sqlalchemy | lib/sqlalchemy/util/compat.py | inspect_formatargspec | 187 | 0.002 | statement_count=29 |
| sqlalchemy | lib/sqlalchemy/sql/compiler.py | SQLCompiler._inserted_primary_key_from_lastrowid_getter | 2322 | 0.003 | statement_count=20 |
| sphinx | sphinx/domains/c/_symbol.py | Symbol.direct_lookup | 657 | 0.003 | statement_count=18 |
| sphinx | sphinx/cmd/build.py | _bug_report_info | 455 | 0.003 | statement_count=15 |
| sphinx | sphinx/theming.py | _validate_theme_toml | 363 | 0.003 | statement_count=16 |
| sphinx | sphinx/builders/html/__init__.py | StandaloneHTMLBuilder.post_process_images | 961 | 0.004 | statement_count=17 |
| starlette | benchmarks/gzip_benchmark.py | test_gzip | 248 | 0.005 | statement_count=16 |
| sqlalchemy | lib/sqlalchemy/orm/query.py | Query._get_options | 471 | 0.005 | statement_count=18 |

### Class-specific failure pattern

- Of 173 false negatives, 142 have statement_count < 20 (within 5 statements of the >=15 threshold -- a near-boundary miss) vs 7 with statement_count >= 25 (comfortably over threshold, a clearer miss, not just a close call).
- Of 229 false positives, 47 have statement_count < 10 -- well under the >=15 threshold, so these are not near-miss boundary cases but methods the model flagged as long despite a short body, likely picking up on graph-structural context (e.g. calling into large surrounding code) rather than the method's own length.

- Errors by repo: sqlalchemy (210), sphinx (176), starlette (16).

## feature_envy

- n=12936 positives=278 precision=0.261 recall=0.532 f1=0.351 (tp=148 fp=418 fn=130 tn=12240)

### Hardest false positives (top 15 by confidence)

| repo | file | qualname | line | prob | metrics |
|---|---|---|---|---|---|
| sphinx | sphinx/transforms/post_transforms/images.py | ImageDownloader.handle | 61 | 1.000 | dominant_external_count=2, self_access_count=3, dominant_external_receiver=CRITICAL_PATH_CHAR_RE |
| sphinx | sphinx/domains/std/__init__.py | StandardDomain.process_doc | 937 | 1.000 | dominant_external_count=4, self_access_count=8, dominant_external_receiver=document |
| sphinx | sphinx/domains/cpp/_ast.py | ASTType.get_id | 3389 | 1.000 | dominant_external_count=9, self_access_count=17, dominant_external_receiver=symbol |
| sphinx | sphinx/util/inventory.py | InventoryFile.dump | 175 | 1.000 | dominant_external_count=3, self_access_count=0, dominant_external_receiver=env |
| sphinx | sphinx/ext/todo.py | TodoDomain.process_doc | 84 | 1.000 | dominant_external_count=2, self_access_count=1, dominant_external_receiver=document |
| sqlalchemy | lib/sqlalchemy/dialects/mssql/base.py | MSDialect._columns_select | 3661 | 0.999 | dominant_external_count=1, self_access_count=1, dominant_external_receiver=ischema.sys_types |
| sphinx | sphinx/ext/todo.py | TodoListProcessor.create_todo_reference | 150 | 0.999 | dominant_external_count=2, self_access_count=2, dominant_external_receiver=todo |
| sphinx | sphinx/ext/extlinks.py | ExternalLinksChecker.check_uri | 61 | 0.999 | dominant_external_count=2, self_access_count=1, dominant_external_receiver=refnode |
| sphinx | sphinx/environment/collectors/metadata.py | MetadataCollector.process_doc | 35 | 0.999 | dominant_external_count=2, self_access_count=0, dominant_external_receiver=doctree |
| sphinx | sphinx/domains/python/__init__.py | PyXRefRole.process_link | 560 | 0.998 | dominant_external_count=2, self_access_count=0, dominant_external_receiver=env |
| sqlalchemy | lib/sqlalchemy/dialects/postgresql/base.py | PGDialect._columns_query | 4294 | 0.998 | dominant_external_count=3, self_access_count=8, dominant_external_receiver=sql.case |
| sphinx | sphinx/domains/c/_symbol.py | Symbol._handle_duplicate_declaration | 530 | 0.998 | dominant_external_count=1, self_access_count=0, dominant_external_receiver=cand_symbol |
| sphinx | sphinx/domains/cpp/_symbol.py | Symbol._handle_duplicate_declaration | 849 | 0.998 | dominant_external_count=1, self_access_count=0, dominant_external_receiver=cand_symbol |
| sphinx | sphinx/builders/_epub_base.py | EpubBuilder.fix_ids | 280 | 0.997 | dominant_external_count=6, self_access_count=11, dominant_external_receiver=tree |
| sphinx | sphinx/transforms/post_transforms/code.py | TrimDoctestFlagsTransform.is_pyconsole | 116 | 0.997 | dominant_external_count=5, self_access_count=0, dominant_external_receiver=node |

### Hardest false negatives (top 15 by confidence)

| repo | file | qualname | line | prob | metrics |
|---|---|---|---|---|---|
| sqlalchemy | lib/sqlalchemy/dialects/postgresql/psycopg.py | PGDialectAsync_psycopg._do_isolation_level | 782 | 0.000 | dominant_external_count=4, self_access_count=0, dominant_external_receiver=connection |
| sphinx | sphinx/ext/autodoc/_dynamic/_mock.py | MockLoader.create_module | 122 | 0.000 | dominant_external_count=3, self_access_count=1, dominant_external_receiver=spec |
| sqlalchemy | lib/sqlalchemy/dialects/mssql/pyodbc.py | _ms_numeric_pyodbc._small_dec_to_string | 413 | 0.000 | dominant_external_count=4, self_access_count=0, dominant_external_receiver=value |
| sqlalchemy | lib/sqlalchemy/engine/default.py | DefaultDialect._set_connection_characteristics | 894 | 0.000 | dominant_external_count=4, self_access_count=2, dominant_external_receiver=connection |
| sqlalchemy | lib/sqlalchemy/orm/identity.py | _WeakInstanceDict.contains_state | 148 | 0.001 | dominant_external_count=3, self_access_count=2, dominant_external_receiver=state |
| sqlalchemy | lib/sqlalchemy/orm/strategy_options.py | _LoadElement.process_compile_state | 1708 | 0.001 | dominant_external_count=4, self_access_count=2, dominant_external_receiver=compile_state |
| sqlalchemy | lib/sqlalchemy/orm/context.py | _MapperEntity._get_entity_clauses | 2891 | 0.001 | dominant_external_count=5, self_access_count=3, dominant_external_receiver=compile_state |
| sphinx | sphinx/application.py | Sphinx.add_node | 975 | 0.001 | dominant_external_count=4, self_access_count=1, dominant_external_receiver=logger |
| sqlalchemy | lib/sqlalchemy/sql/sqltypes.py | SchemaType._variant_mapping_for_set_table | 1210 | 0.002 | dominant_external_count=3, self_access_count=0, dominant_external_receiver=column |
| sqlalchemy | lib/sqlalchemy/dialects/oracle/oracledb.py | OracleDialect_oracledb.do_begin_twophase | 673 | 0.002 | dominant_external_count=3, self_access_count=0, dominant_external_receiver=connection |
| sqlalchemy | lib/sqlalchemy/dialects/mssql/base.py | MSSQLCompiler._check_can_use_fetch_limit | 2164 | 0.002 | dominant_external_count=4, self_access_count=0, dominant_external_receiver=select |
| sqlalchemy | lib/sqlalchemy/dialects/postgresql/base.py | PGCompiler._assert_pg_ts_ext | 2156 | 0.002 | dominant_external_count=3, self_access_count=2, dominant_external_receiver=element |
| sqlalchemy | lib/sqlalchemy/orm/util.py | _ORMJoin._splice_into_center | 1965 | 0.002 | dominant_external_count=6, self_access_count=5, dominant_external_receiver=other |
| sqlalchemy | lib/sqlalchemy/dialects/mssql/base.py | MSSQLCompiler._get_limit_or_fetch | 2140 | 0.002 | dominant_external_count=3, self_access_count=0, dominant_external_receiver=select |
| sqlalchemy | lib/sqlalchemy/dialects/mysql/mysqlconnector.py | MySQLDialect_mysqlconnector.create_connect_args | 165 | 0.003 | dominant_external_count=3, self_access_count=1, dominant_external_receiver=url |

### Class-specific failure pattern

- Of 418 false positives, 304 have dominant_external_count <= self_access_count -- meaning the model flagged them as feature envy even though they fail the label rule's own direction (external access must exceed self access). This is a genuine signal-learning gap, not threshold noise.
- Of 130 false negatives, 57 sit right at the label rule's floor (dominant_external_count==3, the minimum to qualify) vs 33 with dominant_external_count >= 5 -- feature_envy remains the hardest task (rarest class, 278 positives in 12936).

- Errors by repo: sqlalchemy (350), sphinx (193), starlette (5).

## god_class

- n=2612 positives=181 precision=0.864 recall=0.735 f1=0.794 (tp=133 fp=21 fn=48 tn=2410)

### Hardest false positives (top 15 by confidence)

| repo | file | qualname | line | prob | metrics |
|---|---|---|---|---|---|
| sphinx | sphinx/builders/linkcheck.py | HyperlinkAvailabilityCheckWorker | 366 | 0.998 | method_count=7, loc=334 |
| sphinx | sphinx/ext/napoleon/docstring.py | NumpyDocstring | 1116 | 0.976 | method_count=9, loc=322 |
| sphinx | sphinx/ext/inheritance_diagram.py | InheritanceGraph | 146 | 0.965 | method_count=9, loc=237 |
| sqlalchemy | lib/sqlalchemy/event/attr.py | _ClsLevelDispatch | 114 | 0.895 | method_count=10, loc=137 |
| sqlalchemy | lib/sqlalchemy/connectors/asyncio.py | AsyncAdapt_dbapi_cursor | 156 | 0.840 | method_count=22, loc=175 |
| sqlalchemy | lib/sqlalchemy/util/_collections_cy.py | OrderedSet | 81 | 0.815 | method_count=30, loc=177 |
| sqlalchemy | lib/sqlalchemy/util/_collections.py | LRUCache | 453 | 0.796 | method_count=13, loc=103 |
| sqlalchemy | lib/sqlalchemy/engine/result.py | TupleResult | 1426 | 0.792 | method_count=19, loc=161 |
| sphinx | sphinx/pycode/__init__.py | ModuleAnalyzer | 19 | 0.737 | method_count=8, loc=152 |
| sqlalchemy | lib/sqlalchemy/engine/result.py | ResultMetaData | 90 | 0.736 | method_count=17, loc=120 |
| sqlalchemy | lib/sqlalchemy/ext/asyncio/result.py | AsyncTupleResult | 797 | 0.705 | method_count=19, loc=163 |
| sqlalchemy | lib/sqlalchemy/event/attr.py | _CompoundListener | 414 | 0.687 | method_count=12, loc=113 |
| sqlalchemy | lib/sqlalchemy/util/tool_support.py | code_writer_cmd | 36 | 0.665 | method_count=10, loc=167 |
| sqlalchemy | lib/sqlalchemy/engine/result.py | SimpleResultMetaData | 246 | 0.663 | method_count=10, loc=179 |
| sqlalchemy | lib/sqlalchemy/sql/functions.py | GenericFunction | 1498 | 0.646 | method_count=3, loc=164 |

### Hardest false negatives (top 15 by confidence)

| repo | file | qualname | line | prob | metrics |
|---|---|---|---|---|---|
| sphinx | sphinx/directives/code.py | LiteralIncludeReader | 188 | 0.000 | method_count=12, loc=223 |
| sqlalchemy | lib/sqlalchemy/orm/context.py | _ORMCompileState | 386 | 0.000 | method_count=12, loc=303 |
| sqlalchemy | lib/sqlalchemy/sql/selectable.py | Exists | 7185 | 0.000 | method_count=10, loc=241 |
| sphinx | sphinx/domains/cpp/_ast.py | ASTDeclaration | 4500 | 0.000 | method_count=10, loc=224 |
| sqlalchemy | lib/sqlalchemy/dialects/sqlite/base.py | SQLiteDDLCompiler | 1764 | 0.000 | method_count=10, loc=199 |
| sqlalchemy | lib/sqlalchemy/dialects/mssql/base.py | MSDDLCompiler | 2604 | 0.000 | method_count=14, loc=299 |
| sqlalchemy | lib/sqlalchemy/orm/evaluator.py | _EvaluatorCompiler | 60 | 0.000 | method_count=26, loc=306 |
| sqlalchemy | lib/sqlalchemy/dialects/postgresql/pg8000.py | PGDialect_pg8000 | 403 | 0.001 | method_count=20, loc=250 |
| sqlalchemy | lib/sqlalchemy/orm/attributes.py | History | 2347 | 0.001 | method_count=11, loc=199 |
| sqlalchemy | lib/sqlalchemy/sql/compiler.py | GenericTypeCompiler | 7638 | 0.002 | method_count=47, loc=229 |
| sqlalchemy | lib/sqlalchemy/dialects/postgresql/psycopg.py | PGDialect_psycopg | 367 | 0.003 | method_count=22, loc=245 |
| sphinx | sphinx/domains/c/__init__.py | CObject | 109 | 0.004 | method_count=13, loc=230 |
| sphinx | sphinx/writers/text.py | Table | 56 | 0.004 | method_count=14, loc=207 |
| sqlalchemy | lib/sqlalchemy/sql/dml.py | Update | 1633 | 0.005 | method_count=14, loc=195 |
| sqlalchemy | lib/sqlalchemy/dialects/postgresql/psycopg2.py | PGDialect_psycopg2 | 604 | 0.006 | method_count=18, loc=254 |

### Class-specific failure pattern

- Of 21 false positives, 18 satisfy only ONE of the label rule's two AND-conditions (method_count>=10, loc>=192) -- the model appears to respond to the two structural signals somewhat independently rather than strictly requiring both, unlike the rule that generated the label.
- Of 48 false negatives, 13 are within a modest margin of both thresholds (method_count<15, loc<250) -- near-boundary misses rather than large, unambiguous god classes being missed outright.

- Errors by repo: sqlalchemy (52), sphinx (15), starlette (2).
