/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
 *
 * Drop module slots that CPython 3.11 does not know about.
 *
 * Local workaround, not an upstream artifact. The CAM binding is built with
 * CPython 3.12 headers, and its `PyModuleDef` declares
 * `Py_mod_multiple_interpreters` (slot ID 3, introduced in 3.12). CPython
 * 3.11's import machinery rejects the whole module with
 *
 *     SystemError: module umdk_cam_op_lib uses unknown slot ID 3
 *
 * because slot 3 has no meaning there. This shim rewrites that slot to the
 * `{0, NULL}` terminator before the interpreter walks the table, then calls
 * the real implementation. It is a no-op for modules built for 3.11 or older,
 * which never declare that slot.
 *
 * Both entry points are interposed because CPython 3.11 walks the table in
 * `PyModule_FromDefAndSpec*`, and the module may fill its slot array either
 * before or after it calls `PyModuleDef_Init`.
 *
 * Build:
 *     mkdir -p /opt/cam311
 *     gcc -shared -fPIC -O2 -o /opt/cam311/libpyslotfix.so \
 *         tools/itask/cam311_slot_shim.c -ldl
 *
 * Use it by putting it first on the loader path of every process that imports
 * the CAM binding:
 *     export LD_PRELOAD=/opt/cam311/libpyslotfix.so
 * Set `CAM311_SLOT_SHIM_LOG=1` to print one line per rewritten module.
 */

#define _GNU_SOURCE

#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/* Py_mod_multiple_interpreters, added in CPython 3.12. */
#define PY_MOD_SLOT_MULTIPLE_INTERPRETERS 3

struct py_module_def_slot {
    int slot;
    void *value;
};

/* Leading fields of PyModuleDef, which are stable across CPython 3.x. */
struct py_module_def_prefix {
    void *ob_refcnt;
    void *ob_type;
    const char *m_name;
    const char *m_doc;
    long m_size;
    void *m_methods;
    struct py_module_def_slot *m_slots;
};

static void drop_unsupported_slots(struct py_module_def_prefix *def)
{
    struct py_module_def_slot *slot;

    if (def == NULL || def->m_slots == NULL) {
        return;
    }
    for (slot = def->m_slots; slot->slot != 0; slot++) {
        if (slot->slot == PY_MOD_SLOT_MULTIPLE_INTERPRETERS) {
            slot->slot = 0;
            if (getenv("CAM311_SLOT_SHIM_LOG") != NULL) {
                fprintf(
                    stderr,
                    "[cam311_slot_shim] dropped slot %d of module %s\n",
                    PY_MOD_SLOT_MULTIPLE_INTERPRETERS,
                    def->m_name != NULL ? def->m_name : "<unnamed>");
            }
            return;
        }
    }
}

void *PyModuleDef_Init(struct py_module_def_prefix *def)
{
    static void *(*real_init)(struct py_module_def_prefix *) = NULL;

    if (real_init == NULL) {
        real_init = (void *(*)(struct py_module_def_prefix *)) dlsym(
            RTLD_NEXT, "PyModuleDef_Init");
    }
    drop_unsupported_slots(def);
    return real_init(def);
}

void *PyModule_FromDefAndSpec(struct py_module_def_prefix *def, void *spec)
{
    static void *(*real_from_def)(struct py_module_def_prefix *, void *) = NULL;

    if (real_from_def == NULL) {
        real_from_def = (void *(*)(struct py_module_def_prefix *, void *)) dlsym(
            RTLD_NEXT, "PyModule_FromDefAndSpec");
    }
    drop_unsupported_slots(def);
    return real_from_def(def, spec);
}

void *PyModule_FromDefAndSpec2(
    struct py_module_def_prefix *def,
    void *spec,
    int module_api_version)
{
    static void *(*real_from_def2)(
        struct py_module_def_prefix *, void *, int) = NULL;

    if (real_from_def2 == NULL) {
        real_from_def2 = (void *(*)(struct py_module_def_prefix *, void *, int))
            dlsym(RTLD_NEXT, "PyModule_FromDefAndSpec2");
    }
    drop_unsupported_slots(def);
    return real_from_def2(def, spec, module_api_version);
}
