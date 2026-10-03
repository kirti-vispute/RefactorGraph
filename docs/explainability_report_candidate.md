# Explainability Report -- CANDIDATE checkpoint (validation split only)

GNNExplainer applied to the candidate model (models/hybrid_class_pool_tuned_fixed_data/, Design B class_method_pool=True, seed=43), on the hardest false positives/negatives per task from docs/error_analysis_report_candidate.md. The original Phase 12 report (docs/explainability_report.md) targeted models/hybrid_tuned, a different, older, non-pooled architecture -- this is the first GNNExplainer run actually against the model this project has been evaluating since the Design B change. See scripts/explain_candidate.py module docstring for a real receptive-field bug (class_method_pool's pooling edge has method, not class, as its destination) found and fixed before running this, not blindly inherited from scripts/explain.py. TEST split is never loaded by this script.

Each example is explained from a single-graph forward pass; `single_graph_prob` is reported alongside the batched `prob` from error analysis rather than assumed equal (floating-point batching non-associativity) -- a large gap is flagged.

## long_method

### sqlalchemy/lib/sqlalchemy/orm/_orm_constructors.py:relationship (line 1059) -- false positive

- label=0 batched_prob=1.000 single_graph_prob=1.000
- own raw metrics: {'statement_count': 2}
- receptive field: node types ['function', 'module'], edge types ['function->calls->function', 'module->contains->function']
- top contributing edges:
  - 0.000: module:lib/sqlalchemy/orm/_orm_constructors.py --contains--> function:contains_alias
  - 0.000: module:lib/sqlalchemy/orm/_orm_constructors.py --contains--> function:mapped_column
  - 0.000: module:lib/sqlalchemy/orm/_orm_constructors.py --contains--> function:orm_insert_sentinel
  - 0.000: module:lib/sqlalchemy/orm/_orm_constructors.py --contains--> function:column_property
  - 0.000: module:lib/sqlalchemy/orm/_orm_constructors.py --contains--> function:composite
- own structural feature importance:
  - loc: 0.000
  - statement_count: 0.000
  - param_count: 0.000
  - self_access_count: 0.000
  - external_access_count: 0.000
- own CodeBERT embedding importance (mean over 768 dims): 0.000

### sqlalchemy/lib/sqlalchemy/orm/_orm_constructors.py:mapped_column (line 99) -- false positive

- label=0 batched_prob=1.000 single_graph_prob=1.000
- own raw metrics: {'statement_count': 2}
- receptive field: node types ['function', 'module'], edge types ['function->calls->function', 'module->contains->function']
- top contributing edges:
  - 0.812: module:lib/sqlalchemy/orm/_orm_constructors.py --contains--> function:orm_insert_sentinel
  - 0.198: function:orm_insert_sentinel --calls--> function:mapped_column
  - 0.160: module:lib/sqlalchemy/orm/_orm_constructors.py --contains--> function:mapped_column
  - 0.000: module:lib/sqlalchemy/orm/_orm_constructors.py --contains--> function:contains_alias
  - 0.000: module:lib/sqlalchemy/orm/_orm_constructors.py --contains--> function:column_property
- own structural feature importance:
  - loc: 0.612
  - external_access_count: 0.282
  - statement_count: 0.278
  - param_count: 0.270
  - self_access_count: 0.259
- own CodeBERT embedding importance (mean over 768 dims): 0.275

### sqlalchemy/lib/sqlalchemy/sql/selectable.py:HasCTE.cte (line 2714) -- false positive

- label=0 batched_prob=1.000 single_graph_prob=1.000
- own raw metrics: {'statement_count': 2}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'class->inherits->class', 'method->calls->method', 'module->contains->class']
- top contributing edges:
  - 0.797: class:Values --inherits--> class:HasCTE
  - 0.797: class:SelectBase --inherits--> class:HasCTE
  - 0.796: module:lib/sqlalchemy/sql/selectable.py --contains--> class:HasCTE
  - 0.795: class:HasCTE --contains--> method:HasCTE.cte
  - 0.000: class:ExecutableReturnsRows --inherits--> class:ReturnsRows
- own structural feature importance:
  - loc: 0.646
  - statement_count: 0.293
  - self_access_count: 0.288
  - external_access_count: 0.271
  - param_count: 0.220
- own CodeBERT embedding importance (mean over 768 dims): 0.275

### sqlalchemy/lib/sqlalchemy/orm/collections.py:__go (line 1543) -- false negative

- label=1 batched_prob=0.000 single_graph_prob=0.000
- own raw metrics: {'statement_count': 15}
- receptive field: node types ['function', 'module'], edge types ['function->calls->function', 'module->contains->function']
- top contributing edges:
  - 0.209: module:lib/sqlalchemy/orm/collections.py --contains--> function:__go
  - 0.000: module:lib/sqlalchemy/orm/collections.py --contains--> function:collection_adapter
  - 0.000: module:lib/sqlalchemy/orm/collections.py --contains--> function:bulk_replace
  - 0.000: module:lib/sqlalchemy/orm/collections.py --contains--> function:_prepare_instrumentation
  - 0.000: module:lib/sqlalchemy/orm/collections.py --contains--> function:_instrument_class
- own structural feature importance:
  - self_access_count: 0.306
  - param_count: 0.298
  - statement_count: 0.291
  - loc: 0.272
  - external_access_count: 0.243
- own CodeBERT embedding importance (mean over 768 dims): 0.281

### sphinx/sphinx/ext/imgmath.py:setup (line 390) -- false negative

- label=1 batched_prob=0.000 single_graph_prob=0.000
- own raw metrics: {'statement_count': 15}
- receptive field: node types ['function', 'module'], edge types ['function->calls->function', 'module->contains->function']
- top contributing edges:
  - 0.200: module:sphinx/ext/imgmath.py --contains--> function:setup
  - 0.000: module:sphinx/ext/imgmath.py --contains--> function:read_svg_depth
  - 0.000: module:sphinx/ext/imgmath.py --contains--> function:write_svg_depth
  - 0.000: module:sphinx/ext/imgmath.py --contains--> function:generate_latex_macro
  - 0.000: module:sphinx/ext/imgmath.py --contains--> function:compile_math
- own structural feature importance:
  - loc: 0.333
  - statement_count: 0.327
  - external_access_count: 0.290
  - self_access_count: 0.278
  - param_count: 0.267
- own CodeBERT embedding importance (mean over 768 dims): 0.289

### sqlalchemy/lib/sqlalchemy/sql/visitors.py:clone (line 1125) -- false negative

- label=1 batched_prob=0.000 single_graph_prob=0.005
- own raw metrics: {'statement_count': 16}
- receptive field: node types ['class', 'function', 'method', 'module'], edge types ['class->contains->method', 'function->calls->function', 'method->calls->function', 'method->calls->method', 'module->contains->function']
- top contributing edges:
  - 0.799: function:deferred_copy_internals --calls--> function:replacement_traverse
  - 0.209: module:lib/sqlalchemy/sql/visitors.py --contains--> function:replacement_traverse
  - 0.204: method:ReplacingExternalTraversal.traverse --calls--> function:replacement_traverse
  - 0.201: function:replacement_traverse --calls--> function:clone
  - 0.000: module:lib/sqlalchemy/sql/visitors.py --contains--> function:_generate_traversal_dispatch
- own structural feature importance:
  - statement_count: 0.376
  - loc: 0.366
  - param_count: 0.343
  - external_access_count: 0.323
  - self_access_count: 0.289
- own CodeBERT embedding importance (mean over 768 dims): 0.325

## feature_envy

### sqlalchemy/lib/sqlalchemy/dialects/mssql/base.py:MSDialect._columns_select (line 3661) -- false positive

- label=0 batched_prob=1.000 single_graph_prob=1.000
- own raw metrics: {'dominant_external_count': 1, 'self_access_count': 1, 'dominant_external_receiver': 'ischema.sys_types'}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'class->inherits->class', 'method->calls->method', 'module->contains->class']
- top contributing edges:
  - 0.805: method:MSDialect.get_multi_columns --calls--> method:MSDialect._fetch_multi_columns
  - 0.796: class:MSDialect --contains--> method:MSDialect._columns_select
  - 0.790: class:MSDialect --contains--> method:MSDialect._fetch_multi_columns
  - 0.201: method:MSDialect._fetch_multi_columns_temp --calls--> method:MSDialect._fetch_multi_columns
  - 0.200: module:lib/sqlalchemy/dialects/mssql/base.py --contains--> class:MSDialect
- own structural feature importance:
  - external_access_count: 0.649
  - self_access_count: 0.551
  - loc: 0.346
  - param_count: 0.292
  - statement_count: 0.273
- own CodeBERT embedding importance (mean over 768 dims): 0.284

### sqlalchemy/lib/sqlalchemy/dialects/postgresql/base.py:PGDialect._columns_query (line 4294) -- false positive

- label=0 batched_prob=1.000 single_graph_prob=1.000
- own raw metrics: {'dominant_external_count': 3, 'self_access_count': 5, 'dominant_external_receiver': 'sql.case'}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'class->inherits->class', 'method->calls->method', 'module->contains->class']
- top contributing edges:
  - 0.808: class:PGDialect --contains--> method:PGDialect.get_multi_columns
  - 0.797: method:PGDialect.get_multi_columns --calls--> method:PGDialect._columns_query
  - 0.214: class:PGDialect --contains--> method:PGDialect._columns_query
  - 0.203: module:lib/sqlalchemy/dialects/postgresql/base.py --contains--> class:PGDialect
  - 0.000: class:ReflectedDomain --inherits--> class:ReflectedNamedType
- own structural feature importance:
  - external_access_count: 0.612
  - statement_count: 0.287
  - self_access_count: 0.280
  - param_count: 0.268
  - loc: 0.262
- own CodeBERT embedding importance (mean over 768 dims): 0.275

### sqlalchemy/lib/sqlalchemy/pool/base.py:_ConnectionFairy._checkout (line 1263) -- false positive

- label=0 batched_prob=1.000 single_graph_prob=1.000
- own raw metrics: {'dominant_external_count': 6, 'self_access_count': 0, 'dominant_external_receiver': 'pool.logger'}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'class->inherits->class', 'method->calls->method', 'module->contains->class']
- top contributing edges:
  - 0.795: module:lib/sqlalchemy/pool/base.py --contains--> class:_ConnectionFairy
  - 0.207: class:_ConnectionFairy --contains--> method:_ConnectionFairy._checkout
  - 0.000: class:_AsyncConnDialect --inherits--> class:_ConnDialect
  - 0.000: class:ConnectionPoolEntry --inherits--> class:ManagesConnection
  - 0.000: class:_ConnectionRecord --inherits--> class:ConnectionPoolEntry
- own structural feature importance:
  - external_access_count: 0.643
  - self_access_count: 0.569
  - loc: 0.315
  - statement_count: 0.286
  - param_count: 0.253
- own CodeBERT embedding importance (mean over 768 dims): 0.276

### sphinx/sphinx/ext/autodoc/_dynamic/_mock.py:MockLoader.create_module (line 122) -- false negative

- label=1 batched_prob=0.000 single_graph_prob=0.000
- own raw metrics: {'dominant_external_count': 3, 'self_access_count': 0, 'dominant_external_receiver': 'spec'}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'module->contains->class']
- top contributing edges:
  - 0.194: class:MockLoader --contains--> method:MockLoader.create_module
  - 0.192: module:sphinx/ext/autodoc/_dynamic/_mock.py --contains--> class:MockLoader
  - 0.000: module:sphinx/ext/autodoc/_dynamic/_mock.py --contains--> class:_MockObject
  - 0.000: module:sphinx/ext/autodoc/_dynamic/_mock.py --contains--> class:_MockModule
  - 0.000: module:sphinx/ext/autodoc/_dynamic/_mock.py --contains--> class:MockFinder
- own structural feature importance:
  - loc: 0.311
  - self_access_count: 0.294
  - param_count: 0.290
  - statement_count: 0.283
  - external_access_count: 0.244
- own CodeBERT embedding importance (mean over 768 dims): 0.281

### sqlalchemy/lib/sqlalchemy/orm/context.py:FromStatement._compiler_dispatch (line 1020) -- false negative

- label=1 batched_prob=0.000 single_graph_prob=0.000
- own raw metrics: {'dominant_external_count': 3, 'self_access_count': 1, 'dominant_external_receiver': 'compiler'}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'class->inherits->class', 'method->calls->method', 'module->contains->class']
- top contributing edges:
  - 0.801: class:FromStatement --contains--> method:FromStatement._compiler_dispatch
  - 0.204: module:lib/sqlalchemy/orm/context.py --contains--> class:FromStatement
  - 0.000: class:_AutoflushOnlyORMCompileState --inherits--> class:_AbstractORMCompileState
  - 0.000: class:_ORMCompileState --inherits--> class:_AbstractORMCompileState
  - 0.000: class:_DMLBulkInsertReturningColFilter --inherits--> class:_DMLReturningColFilter
- own structural feature importance:
  - self_access_count: 0.299
  - external_access_count: 0.269
  - loc: 0.257
  - statement_count: 0.238
  - param_count: 0.231
- own CodeBERT embedding importance (mean over 768 dims): 0.283

### sphinx/sphinx/registry.py:SphinxComponentRegistry.add_domain (line 198) -- false negative

- label=1 batched_prob=0.000 single_graph_prob=0.000
- own raw metrics: {'dominant_external_count': 3, 'self_access_count': 2, 'dominant_external_receiver': 'domain'}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'method->calls->method', 'module->contains->class']
- top contributing edges:
  - 0.826: class:SphinxComponentRegistry --contains--> method:SphinxComponentRegistry.add_domain
  - 0.176: module:sphinx/registry.py --contains--> class:SphinxComponentRegistry
  - 0.000: class:SphinxComponentRegistry --contains--> method:SphinxComponentRegistry.__init__
  - 0.000: class:SphinxComponentRegistry --contains--> method:SphinxComponentRegistry.autodoc_attrgettrs
  - 0.000: class:SphinxComponentRegistry --contains--> method:SphinxComponentRegistry.add_builder
- own structural feature importance:
  - loc: 0.547
  - external_access_count: 0.363
  - statement_count: 0.327
  - param_count: 0.296
  - self_access_count: 0.288
- own CodeBERT embedding importance (mean over 768 dims): 0.294

## god_class

### sqlalchemy/lib/sqlalchemy/sql/elements.py:AbstractTextClause (line 2328) -- false positive

- label=0 batched_prob=0.999 single_graph_prob=0.973
- own raw metrics: {'method_count': 8, 'loc': 215, 'field_count': 0}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'class->inherits->class', 'method->calls->method', 'module->contains->class']
- top contributing edges:
  - 0.797: class:AbstractTextClause --contains--> method:AbstractTextClause.type
  - 0.792: class:AbstractTextClause --contains--> method:AbstractTextClause.__and__
  - 0.791: module:lib/sqlalchemy/sql/elements.py --contains--> class:AbstractTextClause
  - 0.791: class:AbstractTextClause --contains--> method:AbstractTextClause.bindparams
  - 0.790: class:AbstractTextClause --contains--> method:AbstractTextClause.self_group
- own structural feature importance:
  - loc: 0.578
  - method_count: 0.556
  - field_count: 0.428
- own CodeBERT embedding importance (mean over 768 dims): 0.408

### sqlalchemy/lib/sqlalchemy/ext/orderinglist.py:OrderingList (line 249) -- false positive

- label=0 batched_prob=0.999 single_graph_prob=0.995
- own raw metrics: {'method_count': 15, 'loc': 177, 'field_count': 3}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'method->calls->method', 'module->contains->class']
- top contributing edges:
  - 0.817: method:OrderingList.__setitem__ --calls--> method:OrderingList.__setitem__
  - 0.817: method:OrderingList._order_entity --calls--> method:OrderingList._set_order_value
  - 0.817: class:OrderingList --contains--> method:OrderingList._raw_append
  - 0.814: class:OrderingList --contains--> method:OrderingList._order_entity
  - 0.804: method:OrderingList.append --calls--> method:OrderingList._order_entity
- own structural feature importance:
  - field_count: 0.597
  - loc: 0.560
  - method_count: 0.560
- own CodeBERT embedding importance (mean over 768 dims): 0.402

### sqlalchemy/lib/sqlalchemy/sql/schema.py:Constraint (line 4467) -- false positive

- label=0 batched_prob=0.997 single_graph_prob=0.990
- own raw metrics: {'method_count': 6, 'loc': 113, 'field_count': 8}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'class->inherits->class', 'method->calls->method', 'module->contains->class']
- top contributing edges:
  - 0.809: class:ForeignKeyConstraint --inherits--> class:ColumnCollectionConstraint
  - 0.801: class:UniqueConstraint --inherits--> class:ColumnCollectionConstraint
  - 0.796: class:Constraint --contains--> method:Constraint.copy
  - 0.787: module:lib/sqlalchemy/sql/schema.py --contains--> class:Constraint
  - 0.786: class:Constraint --contains--> method:Constraint._set_parent
- own structural feature importance:
  - loc: 0.597
  - field_count: 0.571
  - method_count: 0.549
- own CodeBERT embedding importance (mean over 768 dims): 0.454

### sqlalchemy/lib/sqlalchemy/orm/context.py:_ORMCompileState (line 386) -- false negative

- label=1 batched_prob=0.000 single_graph_prob=0.025
- own raw metrics: {'method_count': 12, 'loc': 303, 'field_count': 0}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'class->inherits->class', 'method->calls->method', 'module->contains->class']
- top contributing edges:
  - 0.807: class:_ORMCompileState --contains--> method:_ORMCompileState._create_orm_context
  - 0.802: class:_ORMFromStatementCompileState --inherits--> class:_ORMCompileState
  - 0.800: class:_ORMCompileState --contains--> method:_ORMCompileState._append_dedupe_col_collection
  - 0.799: module:lib/sqlalchemy/orm/context.py --contains--> class:_ORMFromStatementCompileState
  - 0.217: class:_ORMCompileState --contains--> method:_ORMCompileState.get_column_descriptions
- own structural feature importance:
  - field_count: 0.485
  - method_count: 0.376
  - loc: 0.331
- own CodeBERT embedding importance (mean over 768 dims): 0.324

### sqlalchemy/lib/sqlalchemy/dialects/postgresql/array.py:ARRAY (line 253) -- false negative

- label=1 batched_prob=0.000 single_graph_prob=0.000
- own raw metrics: {'method_count': 5, 'loc': 249, 'field_count': 4}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'method->calls->method', 'module->contains->class']
- top contributing edges:
  - 0.806: class:ARRAY --contains--> method:ARRAY._against_native_enum
  - 0.208: class:ARRAY --contains--> method:ARRAY.result_processor
  - 0.205: module:lib/sqlalchemy/dialects/postgresql/array.py --contains--> class:ARRAY
  - 0.185: class:ARRAY --contains--> method:ARRAY.literal_processor
  - 0.166: class:ARRAY --contains--> method:ARRAY.__init__
- own structural feature importance:
  - loc: 0.339
  - field_count: 0.316
  - method_count: 0.298
- own CodeBERT embedding importance (mean over 768 dims): 0.290

### sqlalchemy/lib/sqlalchemy/sql/lambdas.py:LambdaElement (line 154) -- false negative

- label=1 batched_prob=0.000 single_graph_prob=0.002
- own raw metrics: {'method_count': 13, 'loc': 271, 'field_count': 8}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'class->inherits->class', 'method->calls->method', 'module->contains->class']
- top contributing edges:
  - 0.824: class:LambdaElement --contains--> method:LambdaElement._retrieve_tracker_rec
  - 0.818: class:LambdaElement --contains--> method:LambdaElement._select_iterable
  - 0.811: class:LambdaElement --contains--> method:LambdaElement._gen_cache_key
  - 0.809: class:LambdaElement --contains--> method:LambdaElement._resolved
  - 0.804: method:LambdaElement._resolved --calls--> method:LambdaElement._setup_binds_for_tracked_expr
- own structural feature importance:
  - method_count: 0.333
  - field_count: 0.325
  - loc: 0.318
- own CodeBERT embedding importance (mean over 768 dims): 0.306
