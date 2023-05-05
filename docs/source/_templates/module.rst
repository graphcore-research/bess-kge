..
   # Copyright (c) 2023 Graphcore Ltd. All rights reserved.
   # Copyright (c) 2007-2023 by the Sphinx team. All rights reserved.

{%- if show_headings %}
{{- [basename, "module"] | join(' ') | e | heading }}

{% endif -%}
.. automodule:: {{ qualname }}
{%- for option in automodule_options %}
   :{{ option }}:
{%- endfor %}

