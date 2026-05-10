.. _modelopt-config-system:

ModelOpt Config System
######################

ModelOpt configs use Python types as the contract and YAML as the portable
data representation. A config file is loaded into ordinary Python
``dict``/``list`` data, optional YAML composition is resolved, and the result is
validated by the owning Pydantic-compatible schema.

The same system is used for reusable YAML snippets under ``modelopt_recipes/``,
recipe loading, and lower-level optimization configs such as PTQ quantization
configs. For the recipe authoring workflow, see :ref:`recipes`.

.. contents:: On this page
   :local:
   :depth: 2


Design goals
============

The config system has four core goals:

* **Typed**: every public config surface has an explicit Python schema.
* **Validated**: type errors, invalid values, and unknown fields fail during
  load or schema construction instead of being silently ignored.
* **Persistent**: resolved configs serialize back to plain data that can be
  written to YAML/JSON or stored in checkpoints.
* **Composable when useful**: YAML authors can factor repeated fragments, such
  as numeric formats or quantizer-entry lists, into shared snippets.

Composability is an authoring convenience, not a replacement for validation.
The resolved data is still checked against the same Python schema that inline
YAML would use.


Main APIs
=========

Use the highest-level loader that matches what you are loading:

``modelopt.recipe.load_recipe(path)``
   Loads a complete model optimization recipe. A recipe contains metadata and
   one or more type-specific sections, such as ``quantize`` for PTQ. The
   returned object is a typed recipe config.

``modelopt.recipe.load_config(path, schema_type=...)``
   Loads one YAML config file or snippet, resolves ``imports`` / ``$import``,
   converts ``eXmY`` floating-point shorthand, and returns plain Python data.
   ``schema_type`` gives the loader enough type context to resolve typed list
   imports when the top-level file does not declare its own schema.

``ModeloptBaseConfig.model_validate(data)``
   Validates plain data against a concrete schema. Config classes inherit from
   :class:`~modelopt.torch.opt.config.ModeloptBaseConfig`, which wraps
   Pydantic's ``BaseModel`` with ModelOpt defaults.

Most users should call ``load_recipe()`` for full recipes. Use
``load_config()`` directly when you are loading a reusable snippet or a
standalone lower-level config.


Schema layer
============

Concrete configs inherit from
:class:`~modelopt.torch.opt.config.ModeloptBaseConfig`. The base class provides
the common behavior expected by ModelOpt configs:

* ``extra="forbid"`` by default, so unknown keys are rejected.
* ``validate_assignment=True``, so changing a field after construction still
  runs validation.
* ``model_dump()`` and ``model_dump_json()`` default to aliases and suppress
  Pydantic serialization warnings.
* Mapping-style access, such as ``cfg["field"]``, ``cfg.get("field")``,
  ``cfg.items()``, and ``cfg.update({...})``, for compatibility with existing
  dict-oriented config code.
* automatic registration with PyTorch safe globals for checkpoint loading with
  ``torch.load(weights_only=True)``.

Some config shapes are better represented as plain data annotations than as
Pydantic models. For example, PTQ ``quant_cfg`` entries are ``TypedDict``
objects and a full quantizer list is a typed list alias. The loader validates
these with Pydantic ``TypeAdapter`` when they are used as reusable snippets.


YAML loading flow
=================

``load_config()`` performs the same high-level steps for every config file:

1. Resolve the file path.
2. Read the optional ``# modelopt-schema: ...`` comment preamble.
3. Parse one YAML document, or two documents for list snippets that also need
   an ``imports`` section.
4. Convert ``eXmY`` strings in ``num_bits`` and ``scale_bits`` fields to
   ``(X, Y)`` tuples.
5. Resolve any ``imports`` declarations and inline ``$import`` references.
6. Validate imported snippets against their declared schemas.
7. Validate the top-level file if it declares ``modelopt-schema``.
8. Return the resolved plain Python ``dict`` or ``list``.

Path resolution for ``load_config()`` checks local filesystem candidates first
and then the built-in ``modelopt_recipes`` package. Suffixes may be omitted;
the loader probes ``.yml`` and ``.yaml``. ``load_recipe()`` uses recipe-oriented
resolution and checks the built-in recipe library before the filesystem for
relative recipe names.


Self-contained configs
======================

A config can be a plain YAML mapping with no composition:

.. code-block:: yaml

   algorithm: max
   quant_cfg:
     - quantizer_name: '*'
       enable: false
     - quantizer_name: '*weight_quantizer'
       cfg:
         num_bits: e2m1
         block_sizes:
           -1: 16
           type: dynamic
           scale_bits: e4m3

This is the baseline representation. YAML stores values, and the Python schema
decides whether those values are valid.


Reusable snippets
=================

Every file referenced from an ``imports`` block is a reusable snippet and must
declare a schema in its initial comment preamble:

.. code-block:: yaml

   # modelopt-schema: modelopt.torch.quantization.config.QuantizerAttributeConfig
   num_bits: e4m3
   axis:

The schema path must resolve under the ``modelopt.`` package and must be a
Pydantic-compatible type, such as a ``ModeloptBaseConfig`` subclass,
``TypedDict``, union, or typed container alias. Snippet schemas are validation
contracts only; they are not arbitrary Python execution hooks.

Schema comments are required for imported snippets because the loader needs to
validate reusable fragments independently and decide how typed list imports
behave. Top-level recipe files usually do not need schema comments because
``load_recipe()`` supplies the schema context.


Composable imports
==================

A YAML file declares its dependencies with a file-local ``imports`` mapping:

.. code-block:: yaml

   imports:
     base_disable_all: configs/ptq/units/base_disable_all
     nvfp4: configs/numerics/nvfp4
     kv_fp8: configs/ptq/units/kv_fp8

References use ``$import`` at the point where the imported data should appear:

.. code-block:: yaml

   algorithm: max
   quant_cfg:
     - $import: base_disable_all
     - quantizer_name: '*weight_quantizer'
       cfg:
         $import: nvfp4
     - $import: kv_fp8

``imports`` names are scoped to the file that declares them. Imported snippets
may have their own ``imports`` blocks, which are resolved recursively. Circular
imports are detected and reported as ``ValueError``.


Dict imports
------------

When ``$import`` appears in a mapping, the imported mapping is copied into the
current mapping. Inline keys then override imported keys at that same mapping
level:

.. code-block:: yaml

   cfg:
     $import: nvfp4
     block_sizes:
       -1: 16
       type: static
       scale_bits: e4m3

Multiple imports are applied in list order, then inline keys are applied last:

.. code-block:: yaml

   cfg:
     $import: [base_format, kv_overrides]
     axis: 0

Dict imports are shallow at the mapping where they appear. If one nested leaf
differs, provide the full nested object inline or create a named snippet for
that variant.


List imports
------------

List imports are schema-directed. For a containing list with schema
``list[T]``:

* importing a snippet with schema ``list[T]`` splices the imported entries into
  the containing list;
* importing a snippet with schema ``T`` appends the imported object as one list
  item;
* importing any other schema raises an error;
* importing into an untyped list raises an error.

Example:

.. code-block:: yaml

   quant_cfg:
     - $import: base_disable_all          # QuantizerCfgEntry, appended
     - quantizer_name: '*weight_quantizer'
       cfg:
         $import: nvfp4                   # QuantizerAttributeConfig, dict import
     - $import: kv_fp8                    # QuantizerCfgListConfig, spliced

A list-entry import must be a mapping whose only key is ``$import``. Put
overrides in the imported snippet or write the entry inline.


Multi-document list snippets
----------------------------

YAML allows only one root node per document. A list-valued snippet that also
needs an ``imports`` block therefore uses two YAML documents: the first document
contains ``imports``, and the second document contains the list payload.

.. code-block:: yaml

   # modelopt-schema: modelopt.torch.quantization.config.QuantizerCfgListConfig
   imports:
     fp8: configs/numerics/fp8
   ---
   - quantizer_name: '*[kv]_bmm_quantizer'
     cfg:
       $import: fp8

The loader resolves imports in the second document and returns the resolved
list.


Recipe integration
==================

Recipes are built on the general config loader. For a single-file PTQ recipe,
``load_recipe()`` calls ``load_config(recipe_file, schema_type=ModelOptPTQRecipe)``
so imports can be resolved with recipe-level type context, then constructs a
``ModelOptPTQRecipe``.

For a directory recipe, the file name supplies the section key:

.. code-block:: text

   my_recipe/
   +-- metadata.yaml
   +-- quantize.yaml

``metadata.yaml`` is loaded with ``RecipeMetadataConfig`` context. For PTQ,
``quantize.yaml`` is loaded with ``QuantizeConfig`` context and then assembled
with the metadata into the final recipe object.


Validation and persistence
==========================

Validation belongs to Python schemas, not to YAML syntax. The loader resolves
YAML into plain data and validates imported snippets, while the owning config
API validates the final object. This keeps persistence simple:

.. code-block:: python

   from modelopt.recipe import load_recipe

   recipe = load_recipe("general/ptq/nvfp4_default-kv_fp8")
   resolved = recipe.model_dump()

``resolved`` is ordinary Python data. It can be serialized with
``yaml.safe_dump()``, written to JSON, compared in tests, or embedded in a
checkpoint. Reloading is the reverse operation: read plain data and validate it
with the appropriate schema.


Authoring guidelines
====================

When adding new config files or snippets:

* Prefer self-contained YAML when the config is used only once.
* Use ``imports`` / ``$import`` when a fragment is shared across recipes or
  when factoring makes review materially clearer.
* Add ``# modelopt-schema: ...`` to every reusable snippet referenced by
  ``imports``.
* Use a concrete typed list schema for list snippets so appending vs splicing is
  unambiguous.
* Keep top-level recipe files free of schema comments unless they are intended
  to be imported as snippets.
* Use ``e4m3``, ``e2m1``, and related ``eXmY`` strings for ``num_bits`` and
  ``scale_bits`` in YAML instead of Python tuple syntax.
* Do not load recipe YAML with a raw YAML parser in application code. Use
  ``load_recipe()`` or ``load_config()`` so imports, schema checks, and format
  conversion are applied consistently.
