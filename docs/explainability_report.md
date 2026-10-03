# Explainability Report (Phase 12 -- validation split only)

GNNExplainer (torch_geometric.explain) applied to the Phase 10 tuned hybrid model (models/hybrid_tuned/), on the hardest false positives/negatives per task from Phase 11 (docs/error_analysis_report.md). TEST split is never loaded by this script. See scripts/explain.py module docstring for why GNNExplainer (not PGExplainer) was used, and for the heterogeneous-graph receptive-field pruning this required.

Each example is explained from a single-graph forward pass (GNNExplainer optimizes a mask per instance), while Phase 11's reported `prob` came from batch_size=64 inference -- `single_graph_prob` below is usually within ~0.01 of `prob` (floating-point batching non-associativity, see Phase 11) but is reported explicitly rather than silently assumed equal; a large gap is flagged.

## feature_envy

### sphinx/sphinx/transforms/post_transforms/images.py:ImageDownloader.handle (line 61) -- false positive

- label=0 batched_prob=1.000 single_graph_prob=1.000
- own raw metrics: {'dominant_external_count': 2, 'self_access_count': 3, 'dominant_external_receiver': 'CRITICAL_PATH_CHAR_RE'}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'class->inherits->class', 'method->calls->method', 'module->contains->class']
- top contributing edges:
  - 0.202: module:sphinx/transforms/post_transforms/images.py --contains--> class:ImageDownloader
  - 0.187: class:ImageDownloader --contains--> method:ImageDownloader.handle
  - 0.000: class:BaseImageConverter --contains--> method:BaseImageConverter.apply
  - 0.000: class:BaseImageConverter --contains--> method:BaseImageConverter.match
  - 0.000: class:BaseImageConverter --contains--> method:BaseImageConverter.handle
- own structural feature importance:
  - external_access_count: 0.632
  - self_access_count: 0.319
  - param_count: 0.302
  - statement_count: 0.273
  - loc: 0.266
- own CodeBERT embedding importance (mean over 768 dims): 0.284

### sphinx/sphinx/domains/std/__init__.py:StandardDomain.process_doc (line 937) -- false positive

- label=0 batched_prob=1.000 single_graph_prob=1.000
- own raw metrics: {'dominant_external_count': 4, 'self_access_count': 8, 'dominant_external_receiver': 'document'}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'class->inherits->class', 'method->calls->method', 'module->contains->class']
- top contributing edges:
  - 0.796: module:sphinx/domains/std/__init__.py --contains--> class:StandardDomain
  - 0.191: class:StandardDomain --contains--> method:StandardDomain.process_doc
  - 0.000: class:GenericObject --contains--> method:GenericObject.handle_signature
  - 0.000: class:GenericObject --contains--> method:GenericObject.add_target_and_index
  - 0.000: class:EnvVarXRefRole --contains--> method:EnvVarXRefRole.result_nodes
- own structural feature importance:
  - external_access_count: 0.648
  - param_count: 0.590
  - self_access_count: 0.368
  - loc: 0.351
  - statement_count: 0.349
- own CodeBERT embedding importance (mean over 768 dims): 0.293

### sphinx/sphinx/domains/cpp/_ast.py:ASTType.get_id (line 3389) -- false positive

- label=0 batched_prob=1.000 single_graph_prob=1.000
- own raw metrics: {'dominant_external_count': 9, 'self_access_count': 17, 'dominant_external_receiver': 'symbol'}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'class->inherits->class', 'method->calls->method', 'module->contains->class']
- top contributing edges:
  - 0.795: module:sphinx/domains/cpp/_ast.py --contains--> class:ASTType
  - 0.793: class:ASTType --contains--> method:ASTType.get_id
  - 0.000: class:ASTIdentifier --contains--> method:ASTIdentifier.__init__
  - 0.000: class:ASTIdentifier --contains--> method:ASTIdentifier.__eq__
  - 0.000: class:ASTIdentifier --contains--> method:ASTIdentifier.__hash__
- own structural feature importance:
  - external_access_count: 0.639
  - param_count: 0.495
  - self_access_count: 0.388
  - statement_count: 0.359
  - loc: 0.312
- own CodeBERT embedding importance (mean over 768 dims): 0.284

### sqlalchemy/lib/sqlalchemy/dialects/postgresql/psycopg.py:PGDialectAsync_psycopg._do_isolation_level (line 782) -- false negative

- label=1 batched_prob=0.000 single_graph_prob=0.000
- own raw metrics: {'dominant_external_count': 4, 'self_access_count': 0, 'dominant_external_receiver': 'connection'}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'class->inherits->class', 'method->calls->method', 'module->contains->class']
- top contributing edges:
  - 0.203: class:PGDialectAsync_psycopg --contains--> method:PGDialectAsync_psycopg._do_isolation_level
  - 0.187: module:lib/sqlalchemy/dialects/postgresql/psycopg.py --contains--> class:PGDialectAsync_psycopg
  - 0.000: class:_PGJSON --contains--> method:_PGJSON.bind_processor
  - 0.000: class:_PGJSONB --contains--> method:_PGJSONB.bind_processor
  - 0.000: class:_PsycopgRange --contains--> method:_PsycopgRange.bind_processor
- own structural feature importance:
  - statement_count: 0.274
  - loc: 0.273
  - self_access_count: 0.272
  - param_count: 0.272
  - external_access_count: 0.270
- own CodeBERT embedding importance (mean over 768 dims): 0.278

### sphinx/sphinx/ext/autodoc/_dynamic/_mock.py:MockLoader.create_module (line 122) -- false negative

- label=1 batched_prob=0.000 single_graph_prob=0.000
- own raw metrics: {'dominant_external_count': 3, 'self_access_count': 1, 'dominant_external_receiver': 'spec'}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'module->contains->class']
- top contributing edges:
  - 0.816: module:sphinx/ext/autodoc/_dynamic/_mock.py --contains--> class:MockLoader
  - 0.180: class:MockLoader --contains--> method:MockLoader.create_module
  - 0.000: class:_MockObject --contains--> method:_MockObject.__new__
  - 0.000: class:_MockObject --contains--> method:_MockObject.__init__
  - 0.000: class:_MockObject --contains--> method:_MockObject.__len__
- own structural feature importance:
  - statement_count: 0.303
  - loc: 0.286
  - self_access_count: 0.273
  - param_count: 0.266
  - external_access_count: 0.264
- own CodeBERT embedding importance (mean over 768 dims): 0.281

### sqlalchemy/lib/sqlalchemy/dialects/mssql/pyodbc.py:_ms_numeric_pyodbc._small_dec_to_string (line 413) -- false negative

- label=1 batched_prob=0.000 single_graph_prob=0.008
- own raw metrics: {'dominant_external_count': 4, 'self_access_count': 0, 'dominant_external_receiver': 'value'}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'class->inherits->class', 'module->contains->class']
- top contributing edges:
  - 0.830: class:_MSNumeric_pyodbc --inherits--> class:_ms_numeric_pyodbc
  - 0.827: class:_ms_numeric_pyodbc --contains--> method:_ms_numeric_pyodbc._small_dec_to_string
  - 0.805: module:lib/sqlalchemy/dialects/mssql/pyodbc.py --contains--> class:_ms_numeric_pyodbc
  - 0.202: class:_MSFloat_pyodbc --inherits--> class:_ms_numeric_pyodbc
  - 0.000: class:_ms_numeric_pyodbc --contains--> method:_ms_numeric_pyodbc.bind_processor
- own structural feature importance:
  - self_access_count: 0.308
  - loc: 0.307
  - external_access_count: 0.294
  - statement_count: 0.291
  - param_count: 0.254
- own CodeBERT embedding importance (mean over 768 dims): 0.285

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

### sqlalchemy/lib/sqlalchemy/sql/selectable.py:HasCTE.cte (line 2714) -- false positive

- label=0 batched_prob=1.000 single_graph_prob=1.000
- own raw metrics: {'statement_count': 2}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'class->inherits->class', 'method->calls->method', 'module->contains->class']
- top contributing edges:
  - 0.794: module:lib/sqlalchemy/sql/selectable.py --contains--> class:HasCTE
  - 0.791: class:SelectBase --inherits--> class:HasCTE
  - 0.211: class:HasCTE --contains--> method:HasCTE.cte
  - 0.207: class:Values --inherits--> class:HasCTE
  - 0.000: class:_JoinTargetProtocol --contains--> method:_JoinTargetProtocol._from_objects
- own structural feature importance:
  - loc: 0.572
  - param_count: 0.290
  - statement_count: 0.283
  - external_access_count: 0.280
  - self_access_count: 0.264
- own CodeBERT embedding importance (mean over 768 dims): 0.274

### sphinx/sphinx/builders/linkcheck.py:CheckExternalLinksBuilder.process_result (line 112) -- false positive

- label=0 batched_prob=1.000 single_graph_prob=1.000
- own raw metrics: {'statement_count': 7}
- receptive field: node types ['class', 'method', 'module'], edge types ['class->contains->method', 'method->calls->method', 'module->contains->class']
- top contributing edges:
  - 0.822: module:sphinx/builders/linkcheck.py --contains--> class:CheckExternalLinksBuilder
  - 0.807: class:CheckExternalLinksBuilder --contains--> method:CheckExternalLinksBuilder.process_result
  - 0.211: class:CheckExternalLinksBuilder --contains--> method:CheckExternalLinksBuilder.finish
  - 0.210: method:CheckExternalLinksBuilder.finish --calls--> method:CheckExternalLinksBuilder.process_result
  - 0.000: class:_SENTINEL_LAR --contains--> method:_SENTINEL_LAR.__repr__
- own structural feature importance:
  - loc: 0.666
  - external_access_count: 0.585
  - self_access_count: 0.581
  - param_count: 0.274
  - statement_count: 0.258
- own CodeBERT embedding importance (mean over 768 dims): 0.274

### sqlalchemy/lib/sqlalchemy/orm/collections.py:__go (line 1543) -- false negative

- label=1 batched_prob=0.000 single_graph_prob=0.000
- own raw metrics: {'statement_count': 15}
- receptive field: node types ['function', 'module'], edge types ['function->calls->function', 'module->contains->function']
- top contributing edges:
  - 0.204: module:lib/sqlalchemy/orm/collections.py --contains--> function:__go
  - 0.000: module:lib/sqlalchemy/orm/collections.py --contains--> function:collection_adapter
  - 0.000: module:lib/sqlalchemy/orm/collections.py --contains--> function:bulk_replace
  - 0.000: module:lib/sqlalchemy/orm/collections.py --contains--> function:_prepare_instrumentation
  - 0.000: module:lib/sqlalchemy/orm/collections.py --contains--> function:_instrument_class
- own structural feature importance:
  - external_access_count: 0.294
  - statement_count: 0.293
  - self_access_count: 0.282
  - param_count: 0.279
  - loc: 0.236
- own CodeBERT embedding importance (mean over 768 dims): 0.278

### sphinx/sphinx/ext/apidoc/_cli.py:_parse_args (line 282) -- false negative

- label=1 batched_prob=0.000 single_graph_prob=0.001
- own raw metrics: {'statement_count': 17}
- receptive field: node types ['function', 'module'], edge types ['function->calls->function', 'module->contains->function']
- top contributing edges:
  - 0.857: module:sphinx/ext/apidoc/_cli.py --contains--> function:_parse_args
  - 0.843: module:sphinx/ext/apidoc/_cli.py --contains--> function:main
  - 0.141: function:main --calls--> function:_parse_args
  - 0.000: module:sphinx/ext/apidoc/_cli.py --contains--> function:get_parser
  - 0.000: module:sphinx/ext/apidoc/_cli.py --contains--> function:_full_quickstart
- own structural feature importance:
  - statement_count: 0.381
  - param_count: 0.368
  - loc: 0.336
  - external_access_count: 0.274
  - self_access_count: 0.268
- own CodeBERT embedding importance (mean over 768 dims): 0.310

### sphinx/sphinx/cmd/build.py:build_main (line 395) -- false negative

- label=1 batched_prob=0.000 single_graph_prob=0.029
- own raw metrics: {'statement_count': 18}
- receptive field: node types ['function', 'module'], edge types ['function->calls->function', 'module->contains->function']
- top contributing edges:
  - 0.813: module:sphinx/cmd/build.py --contains--> function:build_main
  - 0.208: function:main --calls--> function:build_main
  - 0.202: module:sphinx/cmd/build.py --contains--> function:main
  - 0.000: module:sphinx/cmd/build.py --contains--> function:handle_exception
  - 0.000: module:sphinx/cmd/build.py --contains--> function:jobs_argument
- own structural feature importance:
  - statement_count: 0.449
  - loc: 0.435
  - external_access_count: 0.313
  - self_access_count: 0.305
  - param_count: 0.290
- own CodeBERT embedding importance (mean over 768 dims): 0.357

## god_class

### sphinx/sphinx/builders/linkcheck.py:HyperlinkAvailabilityCheckWorker (line 366) -- false positive

- label=0 batched_prob=0.998 single_graph_prob=0.991
- own raw metrics: {'method_count': 7, 'loc': 334}
- receptive field: node types ['class', 'module'], edge types ['module->contains->class']
- top contributing edges:
  - 0.177: module:sphinx/builders/linkcheck.py --contains--> class:HyperlinkAvailabilityCheckWorker
  - 0.000: module:sphinx/builders/linkcheck.py --contains--> class:_Status
  - 0.000: module:sphinx/builders/linkcheck.py --contains--> class:_SENTINEL_LAR
  - 0.000: module:sphinx/builders/linkcheck.py --contains--> class:CheckExternalLinksBuilder
  - 0.000: module:sphinx/builders/linkcheck.py --contains--> class:HyperlinkCollector
- own structural feature importance:
  - method_count: 0.570
  - field_count: 0.561
  - loc: 0.550
- own CodeBERT embedding importance (mean over 768 dims): 0.350

### sphinx/sphinx/ext/napoleon/docstring.py:NumpyDocstring (line 1116) -- false positive

- label=0 batched_prob=0.976 single_graph_prob=0.907 **(gap=0.069, flagged)**
- own raw metrics: {'method_count': 9, 'loc': 322}
- receptive field: node types ['class', 'module'], edge types ['class->inherits->class', 'module->contains->class']
- top contributing edges:
  - 0.200: module:sphinx/ext/napoleon/docstring.py --contains--> class:NumpyDocstring
  - 0.000: class:NumpyDocstring --inherits--> class:GoogleDocstring
  - 0.000: module:sphinx/ext/napoleon/docstring.py --contains--> class:Deque
  - 0.000: module:sphinx/ext/napoleon/docstring.py --contains--> class:GoogleDocstring
- own structural feature importance:
  - loc: 0.621
  - method_count: 0.575
  - field_count: 0.390
- own CodeBERT embedding importance (mean over 768 dims): 0.398

### sphinx/sphinx/ext/inheritance_diagram.py:InheritanceGraph (line 146) -- false positive

- label=0 batched_prob=0.965 single_graph_prob=0.935
- own raw metrics: {'method_count': 9, 'loc': 237}
- receptive field: node types ['class', 'module'], edge types ['module->contains->class']
- top contributing edges:
  - 0.108: module:sphinx/ext/inheritance_diagram.py --contains--> class:InheritanceGraph
  - 0.000: module:sphinx/ext/inheritance_diagram.py --contains--> class:InheritanceException
  - 0.000: module:sphinx/ext/inheritance_diagram.py --contains--> class:inheritance_diagram
  - 0.000: module:sphinx/ext/inheritance_diagram.py --contains--> class:InheritanceDiagram
- own structural feature importance:
  - method_count: 0.590
  - loc: 0.572
  - field_count: 0.513
- own CodeBERT embedding importance (mean over 768 dims): 0.380

### sphinx/sphinx/directives/code.py:LiteralIncludeReader (line 188) -- false negative

- label=1 batched_prob=0.000 single_graph_prob=0.000
- own raw metrics: {'method_count': 12, 'loc': 223}
- receptive field: node types ['class', 'module'], edge types ['module->contains->class']
- top contributing edges:
  - 0.149: module:sphinx/directives/code.py --contains--> class:LiteralIncludeReader
  - 0.000: module:sphinx/directives/code.py --contains--> class:Highlight
  - 0.000: module:sphinx/directives/code.py --contains--> class:CodeBlock
  - 0.000: module:sphinx/directives/code.py --contains--> class:LiteralInclude
- own structural feature importance:
  - field_count: 0.290
  - method_count: 0.273
  - loc: 0.234
- own CodeBERT embedding importance (mean over 768 dims): 0.279

### sqlalchemy/lib/sqlalchemy/orm/context.py:_ORMCompileState (line 386) -- false negative

- label=1 batched_prob=0.000 single_graph_prob=0.078 **(gap=0.078, flagged)**
- own raw metrics: {'method_count': 12, 'loc': 303}
- receptive field: node types ['class', 'module'], edge types ['class->inherits->class', 'module->contains->class']
- top contributing edges:
  - 0.859: module:lib/sqlalchemy/orm/context.py --contains--> class:_ORMFromStatementCompileState
  - 0.801: module:lib/sqlalchemy/orm/context.py --contains--> class:_ORMSelectCompileState
  - 0.796: module:lib/sqlalchemy/orm/context.py --contains--> class:_ORMCompileState
  - 0.787: class:_ORMFromStatementCompileState --inherits--> class:_ORMCompileState
  - 0.146: class:_ORMSelectCompileState --inherits--> class:_ORMCompileState
- own structural feature importance:
  - field_count: 0.520
  - loc: 0.443
  - method_count: 0.379
- own CodeBERT embedding importance (mean over 768 dims): 0.356

### sqlalchemy/lib/sqlalchemy/sql/selectable.py:Exists (line 7185) -- false negative

- label=1 batched_prob=0.000 single_graph_prob=0.001
- own raw metrics: {'method_count': 10, 'loc': 241}
- receptive field: node types ['class', 'module'], edge types ['class->inherits->class', 'module->contains->class']
- top contributing edges:
  - 0.801: module:lib/sqlalchemy/sql/selectable.py --contains--> class:Exists
  - 0.000: class:ExecutableReturnsRows --inherits--> class:ReturnsRows
  - 0.000: class:TypedReturnsRows --inherits--> class:ExecutableReturnsRows
  - 0.000: class:Selectable --inherits--> class:ReturnsRows
  - 0.000: class:FromClause --inherits--> class:Selectable
- own structural feature importance:
  - field_count: 0.606
  - method_count: 0.280
  - loc: 0.251
- own CodeBERT embedding importance (mean over 768 dims): 0.285
