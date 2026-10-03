# Error Analysis Report -- CANDIDATE checkpoint (validation split only)

Model: candidate (models/hybrid_class_pool_tuned_fixed_data/, Design B class_method_pool=True, seed=43 -- see docs/godclass_formula_revision.md sections 9-11 for how this checkpoint was selected). Same methodology as the original Phase 11 report (docs/error_analysis_report.md), but that report was for models/hybrid_tuned, a different (non-pooled) architecture -- this is the first error analysis actually run against the model this project has been evaluating since the Design B change. TEST split is never loaded by this script.

False positive (FP) = predicted smelly, label says not smelly. False negative (FN) = predicted not smelly, label says smelly.

## long_method

- n=15576 positives=1205 precision=0.727 recall=0.904 f1=0.806 (tp=1089 fp=408 fn=116 tn=13963)

### Hardest false positives (top 15 by confidence)

| repo | file | qualname | line | prob | metrics |
|---|---|---|---|---|---|
| sqlalchemy | lib/sqlalchemy/orm/_orm_constructors.py | relationship | 1059 | 1.000 | statement_count=2 |
| sqlalchemy | lib/sqlalchemy/orm/_orm_constructors.py | mapped_column | 99 | 1.000 | statement_count=2 |
| sqlalchemy | lib/sqlalchemy/sql/selectable.py | HasCTE.cte | 2714 | 1.000 | statement_count=2 |
| sqlalchemy | lib/sqlalchemy/dialects/postgresql/base.py | PGDialect._columns_query | 4294 | 1.000 | statement_count=12 |
| sqlalchemy | lib/sqlalchemy/engine/base.py | Connection.execution_options | 261 | 1.000 | statement_count=6 |
| sqlalchemy | lib/sqlalchemy/engine/cursor.py | CursorResultMetaData._merge_cursor_description | 543 | 1.000 | statement_count=14 |
| sphinx | sphinx/environment/collectors/toctree.py | TocTreeCollector.process_doc | 64 | 1.000 | statement_count=9 |
| sqlalchemy | lib/sqlalchemy/orm/relationships.py | _JoinCondition._determine_joins | 2466 | 1.000 | statement_count=14 |
| sqlalchemy | lib/sqlalchemy/sql/_elements_constructors.py | bindparam | 553 | 1.000 | statement_count=2 |
| sqlalchemy | lib/sqlalchemy/sql/ddl.py | SchemaDropper.visit_metadata | 1576 | 0.999 | statement_count=14 |
| sqlalchemy | tools/generate_proxy_methods.py | process_class | 178 | 0.999 | statement_count=13 |
| sqlalchemy | lib/sqlalchemy/dialects/mssql/base.py | MSSQLCompiler._render_json_extract_from_binary | 2489 | 0.999 | statement_count=13 |
| sqlalchemy | lib/sqlalchemy/orm/attributes.py | _backref_listeners | 2133 | 0.998 | statement_count=11 |
| sphinx | sphinx/builders/linkcheck.py | CheckExternalLinksBuilder.process_result | 112 | 0.998 | statement_count=7 |
| sqlalchemy | lib/sqlalchemy/orm/relationships.py | _JoinCondition._warn_for_conflicting_sync_targets | 3218 | 0.998 | statement_count=13 |

### Hardest false negatives (top 15 by confidence)

| repo | file | qualname | line | prob | metrics |
|---|---|---|---|---|---|
| sqlalchemy | lib/sqlalchemy/orm/collections.py | __go | 1543 | 0.000 | statement_count=15 |
| sphinx | sphinx/ext/imgmath.py | setup | 390 | 0.000 | statement_count=15 |
| sqlalchemy | lib/sqlalchemy/sql/visitors.py | clone | 1125 | 0.000 | statement_count=16 |
| sphinx | sphinx/ext/autodoc/_legacy_class_based/_documenters.py | is_filtered_inherited_member | 524 | 0.001 | statement_count=15 |
| sphinx | sphinx/transforms/__init__.py | setup | 518 | 0.001 | statement_count=16 |
| sqlalchemy | lib/sqlalchemy/orm/loading.py | polymorphic_instance | 1514 | 0.002 | statement_count=15 |
| sqlalchemy | lib/sqlalchemy/inspection.py | inspect | 114 | 0.002 | statement_count=16 |
| sqlalchemy | lib/sqlalchemy/orm/descriptor_props.py | fset | 331 | 0.002 | statement_count=16 |
| sphinx | sphinx/cmd/build.py | _bug_report_info | 455 | 0.003 | statement_count=15 |
| sphinx | sphinx/ext/apidoc/_cli.py | _parse_args | 282 | 0.007 | statement_count=17 |
| sphinx | sphinx/theming.py | _validate_theme_toml | 363 | 0.008 | statement_count=16 |
| sqlalchemy | lib/sqlalchemy/util/langhelpers.py | load_uncompiled_module | 2388 | 0.008 | statement_count=17 |
| sphinx | sphinx/util/docfields.py | handle_item | 295 | 0.010 | statement_count=15 |
| sphinx | sphinx/ext/autodoc/__init__.py | _register_directives | 215 | 0.012 | statement_count=19 |
| sqlalchemy | lib/sqlalchemy/orm/collections.py | wrapper | 985 | 0.013 | statement_count=22 |

### Class-specific failure pattern

- Of 116 false negatives, 102 have statement_count < 20 (near-boundary miss) vs 2 with statement_count >= 25 (comfortably over threshold, a clearer miss).
- Of 408 false positives, 92 have statement_count < 10 -- not near-miss boundary cases.

- Errors by repo: sqlalchemy (313), sphinx (195), starlette (16).

## feature_envy

- n=12966 positives=259 precision=0.374 recall=0.653 f1=0.475 (tp=169 fp=283 fn=90 tn=12424)

### Hardest false positives (top 15 by confidence)

| repo | file | qualname | line | prob | metrics |
|---|---|---|---|---|---|
| sqlalchemy | lib/sqlalchemy/dialects/mssql/base.py | MSDialect._columns_select | 3661 | 1.000 | dominant_external_count=1, self_access_count=1, dominant_external_receiver=ischema.sys_types |
| sqlalchemy | lib/sqlalchemy/dialects/postgresql/base.py | PGDialect._columns_query | 4294 | 1.000 | dominant_external_count=3, self_access_count=5, dominant_external_receiver=sql.case |
| sqlalchemy | lib/sqlalchemy/pool/base.py | _ConnectionFairy._checkout | 1263 | 1.000 | dominant_external_count=6, self_access_count=0, dominant_external_receiver=pool.logger |
| sqlalchemy | lib/sqlalchemy/dialects/postgresql/base.py | PGDialect._domain_query | 5731 | 1.000 | dominant_external_count=2, self_access_count=1, dominant_external_receiver=sql.func.array_agg |
| sqlalchemy | lib/sqlalchemy/dialects/oracle/base.py | OracleDialect._constraint_query | 3397 | 1.000 | dominant_external_count=2, self_access_count=0, dominant_external_receiver=dictionary.all_cons_columns |
| sphinx | sphinx/domains/cpp/_ast.py | ASTTypeWithInit.get_id | 3550 | 1.000 | dominant_external_count=2, self_access_count=0, dominant_external_receiver=self.type |
| sqlalchemy | lib/sqlalchemy/orm/strategies.py | _JoinedLoader._create_eager_join | 2456 | 0.999 | dominant_external_count=9, self_access_count=13, dominant_external_receiver=compile_state |
| sphinx | sphinx/environment/collectors/asset.py | DownloadFileCollector.process_doc | 152 | 0.999 | dominant_external_count=2, self_access_count=0, dominant_external_receiver=app.env |
| sqlalchemy | lib/sqlalchemy/orm/bulk_persistence.py | _BulkORMUpdate._do_post_synchronize_fetch | 1802 | 0.999 | dominant_external_count=5, self_access_count=0, dominant_external_receiver=update_options |
| sqlalchemy | lib/sqlalchemy/sql/ddl.py | SchemaDropper._can_drop_index | 1664 | 0.999 | dominant_external_count=2, self_access_count=2, dominant_external_receiver=index |
| sqlalchemy | lib/sqlalchemy/sql/ddl.py | SchemaGenerator._can_create_index | 1420 | 0.999 | dominant_external_count=2, self_access_count=2, dominant_external_receiver=index |
| sqlalchemy | lib/sqlalchemy/dialects/mssql/base.py | MSDialect._table_comment_select | 4739 | 0.999 | dominant_external_count=1, self_access_count=0, dominant_external_receiver=sql.select.select_from.join.outerjoin |
| sqlalchemy | lib/sqlalchemy/orm/strategies.py | _SubqueryLoader._generate_from_original_query | 1550 | 0.998 | dominant_external_count=2, self_access_count=0, dominant_external_receiver=orig_compile_state |
| sphinx | sphinx/application.py | Sphinx.build | 434 | 0.997 | dominant_external_count=6, self_access_count=13, dominant_external_receiver=logger |
| sqlalchemy | lib/sqlalchemy/dialects/mssql/base.py | MSDDLCompiler.visit_primary_key_constraint | 2777 | 0.997 | dominant_external_count=2, self_access_count=1, dominant_external_receiver=constraint |

### Hardest false negatives (top 15 by confidence)

| repo | file | qualname | line | prob | metrics |
|---|---|---|---|---|---|
| sphinx | sphinx/ext/autodoc/_dynamic/_mock.py | MockLoader.create_module | 122 | 0.000 | dominant_external_count=3, self_access_count=0, dominant_external_receiver=spec |
| sqlalchemy | lib/sqlalchemy/orm/context.py | FromStatement._compiler_dispatch | 1020 | 0.000 | dominant_external_count=3, self_access_count=1, dominant_external_receiver=compiler |
| sphinx | sphinx/registry.py | SphinxComponentRegistry.add_domain | 198 | 0.000 | dominant_external_count=3, self_access_count=2, dominant_external_receiver=domain |
| sqlalchemy | lib/sqlalchemy/pool/impl.py | SingletonThreadPool._transfer_from | 382 | 0.000 | dominant_external_count=3, self_access_count=2, dominant_external_receiver=other_singleton_pool |
| sphinx | sphinx/directives/other.py | TocTree.parse_content | 89 | 0.001 | dominant_external_count=3, self_access_count=2, dominant_external_receiver=logger |
| sqlalchemy | lib/sqlalchemy/dialects/sqlite/base.py | SQLiteCompiler.visit_extract | 1533 | 0.001 | dominant_external_count=3, self_access_count=2, dominant_external_receiver=extract |
| sqlalchemy | lib/sqlalchemy/sql/sqltypes.py | JSON.bind_processor | 2970 | 0.001 | dominant_external_count=3, self_access_count=1, dominant_external_receiver=dialect |
| sphinx | sphinx/registry.py | SphinxComponentRegistry.get_translator_class | 413 | 0.001 | dominant_external_count=3, self_access_count=1, dominant_external_receiver=builder |
| sqlalchemy | lib/sqlalchemy/util/concurrency.py | _Runner.close | 262 | 0.002 | dominant_external_count=3, self_access_count=2, dominant_external_receiver=self._loop |
| sphinx | sphinx/application.py | Sphinx._init_i18n | 355 | 0.002 | dominant_external_count=3, self_access_count=2, dominant_external_receiver=logger |
| sqlalchemy | lib/sqlalchemy/orm/clsregistry.py | _MultipleClassMarker.add_item | 240 | 0.002 | dominant_external_count=3, self_access_count=2, dominant_external_receiver=item |
| sqlalchemy | lib/sqlalchemy/sql/lambdas.py | AnalyzedCode._cache_key_getter_tracked_literal | 1060 | 0.003 | dominant_external_count=3, self_access_count=1, dominant_external_receiver=pytracker |
| sqlalchemy | lib/sqlalchemy/orm/decl_api.py | registry._add_manager | 1467 | 0.003 | dominant_external_count=4, self_access_count=1, dominant_external_receiver=manager |
| sqlalchemy | lib/sqlalchemy/dialects/oracle/base.py | OracleTypeCompiler.visit_INTERVAL | 1148 | 0.003 | dominant_external_count=4, self_access_count=0, dominant_external_receiver=type_ |
| sqlalchemy | lib/sqlalchemy/dialects/mssql/base.py | MSSQLCompiler.visit_extract | 2311 | 0.006 | dominant_external_count=3, self_access_count=1, dominant_external_receiver=extract |

### Class-specific failure pattern

- Of 283 false positives, 151 have dominant_external_count <= self_access_count -- flagged despite failing the label rule's own direction.
- Of 90 false negatives, 58 sit right at the rule's floor (dominant_external_count==3) vs 12 with dominant_external_count >= 5 -- still the hardest task (259 positives in 12966).

- Errors by repo: sqlalchemy (248), sphinx (118), starlette (7).

## god_class

- n=2612 positives=208 precision=0.776 recall=0.817 f1=0.796 (tp=170 fp=49 fn=38 tn=2355)

### Hardest false positives (top 15 by confidence)

| repo | file | qualname | line | prob | metrics |
|---|---|---|---|---|---|
| sqlalchemy | lib/sqlalchemy/sql/elements.py | AbstractTextClause | 2328 | 0.999 | method_count=8, loc=215, field_count=0 |
| sqlalchemy | lib/sqlalchemy/ext/orderinglist.py | OrderingList | 249 | 0.999 | method_count=15, loc=177, field_count=3 |
| sqlalchemy | lib/sqlalchemy/sql/schema.py | Constraint | 4467 | 0.997 | method_count=6, loc=113, field_count=8 |
| sqlalchemy | lib/sqlalchemy/event/attr.py | _ClsLevelDispatch | 114 | 0.996 | method_count=10, loc=137, field_count=6 |
| sqlalchemy | lib/sqlalchemy/event/registry.py | _EventKey | 220 | 0.991 | method_count=12, loc=172, field_count=6 |
| sqlalchemy | lib/sqlalchemy/sql/selectable.py | TableClause | 3155 | 0.986 | method_count=13, loc=172, field_count=6 |
| sqlalchemy | lib/sqlalchemy/sql/lambdas.py | AnalyzedFunction | 1098 | 0.980 | method_count=4, loc=178, field_count=9 |
| sqlalchemy | lib/sqlalchemy/engine/result.py | FilterResult | 1190 | 0.974 | method_count=12, loc=101, field_count=1 |
| sqlalchemy | lib/sqlalchemy/orm/path_registry.py | _PropRegistry | 599 | 0.963 | method_count=4, loc=138, field_count=11 |
| sqlalchemy | lib/sqlalchemy/engine/result.py | TupleResult | 1426 | 0.948 | method_count=19, loc=161, field_count=0 |
| sqlalchemy | lib/sqlalchemy/connectors/asyncio.py | AsyncAdapt_dbapi_cursor | 156 | 0.942 | method_count=22, loc=175, field_count=5 |
| sqlalchemy | lib/sqlalchemy/engine/result.py | SimpleResultMetaData | 246 | 0.942 | method_count=10, loc=179, field_count=8 |
| sqlalchemy | lib/sqlalchemy/sql/selectable.py | TextualSelect | 7428 | 0.939 | method_count=9, loc=158, field_count=3 |
| sqlalchemy | lib/sqlalchemy/orm/descriptor_props.py | SynonymProperty | 995 | 0.937 | method_count=8, loc=171, field_count=7 |
| sqlalchemy | lib/sqlalchemy/ext/asyncio/result.py | AsyncTupleResult | 797 | 0.936 | method_count=19, loc=163, field_count=0 |

### Hardest false negatives (top 15 by confidence)

| repo | file | qualname | line | prob | metrics |
|---|---|---|---|---|---|
| sqlalchemy | lib/sqlalchemy/orm/context.py | _ORMCompileState | 386 | 0.000 | method_count=12, loc=303, field_count=0 |
| sqlalchemy | lib/sqlalchemy/dialects/postgresql/array.py | ARRAY | 253 | 0.000 | method_count=5, loc=249, field_count=4 |
| sqlalchemy | lib/sqlalchemy/sql/lambdas.py | LambdaElement | 154 | 0.000 | method_count=13, loc=271, field_count=8 |
| sqlalchemy | lib/sqlalchemy/sql/elements.py | TextClause | 2545 | 0.000 | method_count=6, loc=192, field_count=2 |
| sqlalchemy | lib/sqlalchemy/dialects/sqlite/base.py | SQLiteDDLCompiler | 1764 | 0.001 | method_count=10, loc=199, field_count=0 |
| sqlalchemy | lib/sqlalchemy/orm/bulk_persistence.py | _BulkORMDelete | 1945 | 0.004 | method_count=5, loc=224, field_count=2 |
| sqlalchemy | lib/sqlalchemy/orm/attributes.py | History | 2347 | 0.004 | method_count=11, loc=199, field_count=0 |
| sphinx | sphinx/directives/code.py | LiteralIncludeReader | 188 | 0.008 | method_count=12, loc=223, field_count=4 |
| sqlalchemy | lib/sqlalchemy/sql/elements.py | BindParameter | 2017 | 0.008 | method_count=12, loc=291, field_count=13 |
| sqlalchemy | lib/sqlalchemy/dialects/postgresql/hstore.py | HSTORE | 30 | 0.010 | method_count=3, loc=206, field_count=1 |
| sqlalchemy | lib/sqlalchemy/orm/path_registry.py | PathRegistry | 128 | 0.010 | method_count=32, loc=297, field_count=0 |
| sqlalchemy | lib/sqlalchemy/orm/util.py | Bundle | 1565 | 0.010 | method_count=10, loc=225, field_count=6 |
| sqlalchemy | lib/sqlalchemy/orm/evaluator.py | _EvaluatorCompiler | 60 | 0.019 | method_count=26, loc=306, field_count=1 |
| sqlalchemy | lib/sqlalchemy/sql/dml.py | Update | 1633 | 0.027 | method_count=14, loc=195, field_count=2 |
| sqlalchemy | lib/sqlalchemy/orm/strategy_options.py | _AttributeStrategyLoad | 1907 | 0.028 | method_count=7, loc=271, field_count=3 |

### Class-specific failure pattern

- Of 49 false positives: 0 satisfy both rule branches, 0 the size branch only, 0 the field branch only, 49 satisfy NEITHER branch (a genuine signal-learning gap, not just a formula-boundary artifact).
- Of 38 false negatives, 20 are within a modest margin of both thresholds (method_count<15, loc<250) -- near-boundary misses.

- Errors by repo: sqlalchemy (75), sphinx (8), starlette (4).
