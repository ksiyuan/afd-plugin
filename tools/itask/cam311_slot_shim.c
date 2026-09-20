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
 * so this shim rewrites that slot to the `{0, NULL}` terminator from
 * `PyModuleDef_Init` and `PyModule_FromDefAndSpec*`, then calls the real
 * implementation. It is a no-op for modules built for 3.11.
 *
 * The real symbols are resolved defensively: an interpreter whose symbols live
 * in the main executable (a static libpython build) is not reachable through
 * `RTLD_NEXT`, and calling a NULL pointer there is a segfault. When
 * `CAM311_SLOT_SHIM_LOG` is set the shim reports every call it sees, which is
 * how to tell "interposition did not happen" from "the module crashed later".
 *
 * Build:
 *     mkdir -p /opt/cam311
 *     gcc -shared -fPIC -O2 -o /opt/cam311/libpyslotfix.so \
 *         tools/itask/cam311_slot_shim.c -ldl
 *
 * Use:
 *     export LD_PRELOAD=/opt/cam311/libpyslotfix.so
 *     export CAM311_SLOT_SHIM_LOG=1
 */

#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdarg.h>
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

static void log_msg(const char *format, ...)
{
    va_list args;

    if (getenv("CAM311_SLOT_SHIM_LOG") == NULL) {
        return;
    }
    fprintf(stderr, "[cam311_slot_shim] ");
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fprintf(stderr, "\n");
}

static const char *module_name(const struct py_module_def_prefix *def)
{
    if (def == NULL || def->m_name == NULL) {
        return "<unnamed>";
    }
    return def->m_name;
}

/* Resolve `name` without ever returning one of our own definitions. */
static void *resolve_real(const char *name, void *self)
{
    void *symbol = dlsym(RTLD_NEXT, name);

    if (symbol == NULL) {
        void *global = dlopen(NULL, RTLD_LAZY);

        if (global != NULL) {
            symbol = dlsym(global, name);
        }
    }
    if (symbol == self) {
        symbol = NULL;
    }
    if (symbol == NULL) {
        log_msg("cannot resolve %s: interposition is not usable here", name);
    }
    return symbol;
}

static void drop_unsupported_slots(struct py_module_def_prefix *def)
{
    struct py_module_def_slot *slot;

    if (def == NULL || def->m_slots == NULL) {
        return;
    }
    for (slot = def->m_slots; slot->slot != 0; slot++) {
        if (slot->slot == PY_MOD_SLOT_MULTIPLE_INTERPRETERS) {
            slot->slot = 0;
            log_msg(
                "dropped slot %d of module %s (rewrote it to a terminator)",
                PY_MOD_SLOT_MULTIPLE_INTERPRETERS,
                module_name(def));
            return;
        }
    }
}

void *PyModuleDef_Init(struct py_module_def_prefix *def)
{
    static void *(*real_init)(struct py_module_def_prefix *) = NULL;

    if (real_init == NULL) {
        real_init = (void *(*)(struct py_module_def_prefix *)) resolve_real(
            "PyModuleDef_Init",
            (void *) PyModuleDef_Init);
    }
    log_msg("PyModuleDef_Init(%s)", module_name(def));
    if (real_init == NULL) {
        return NULL;
    }
    drop_unsupported_slots(def);
    return real_init(def);
}

void *PyModule_FromDefAndSpec(struct py_module_def_prefix *def, void *spec)
{
    static void *(*real_from_def)(struct py_module_def_prefix *, void *) = NULL;

    if (real_from_def == NULL) {
        real_from_def = (void *(*)(struct py_module_def_prefix *, void *))
            resolve_real(
                "PyModule_FromDefAndSpec",
                (void *) PyModule_FromDefAndSpec);
    }
    log_msg("PyModule_FromDefAndSpec(%s)", module_name(def));
    if (real_from_def == NULL) {
        return NULL;
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
        real_from_def2 =
            (void *(*)(struct py_module_def_prefix *, void *, int)) resolve_real(
                "PyModule_FromDefAndSpec2",
                (void *) PyModule_FromDefAndSpec2);
    }
    log_msg("PyModule_FromDefAndSpec2(%s)", module_name(def));
    if (real_from_def2 == NULL) {
        return NULL;
    }
    drop_unsupported_slots(def);
    return real_from_def2(def, spec, module_api_version);
}
