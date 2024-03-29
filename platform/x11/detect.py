import os
import platform
import sys
from methods import get_compiler_version, using_gcc, using_clang


def is_active():
    return True


def get_name():
    return "X11"


def can_build():
    if os.name != "posix" or sys.platform == "darwin":
        return False

    return True


def get_opts():
    from SCons.Variables import BoolVariable, EnumVariable

    return [
        EnumVariable("linker", "Linker program", "default", ("default", "bfd", "gold", "lld", "mold")),
        BoolVariable("use_llvm", "Use the LLVM compiler", False),
        BoolVariable("use_static_cpp", "Link libgcc and libstdc++ statically for better portability", True),
        BoolVariable('use_coverage', 'Test Pandemonium coverage', False),
        BoolVariable("use_ubsan", "Use LLVM/GCC compiler undefined behavior sanitizer (UBSAN)", False),
        BoolVariable("use_asan", "Use LLVM/GCC compiler address sanitizer (ASAN))", False),
        BoolVariable("use_lsan", "Use LLVM/GCC compiler leak sanitizer (LSAN))", False),
        BoolVariable("use_tsan", "Use LLVM/GCC compiler thread sanitizer (TSAN))", False),
        BoolVariable("use_msan", "Use LLVM/GCC compiler memory sanitizer (MSAN))", False),
        BoolVariable("debug_symbols", "Add debugging symbols to release/release_debug builds", True),
        BoolVariable("separate_debug_symbols", "Create a separate file containing debugging symbols", False),
        BoolVariable("execinfo", "Use libexecinfo on systems where glibc is not available", False),
    ]


def get_flags():
    return []


def configure(env):
    ## Build type

    if env["target"] == "release":
        if env["optimize"] == "speed":  # optimize for speed (default)
            env.Prepend(CCFLAGS=["-O3"])
        elif env["optimize"] == "size":  # optimize for size
            env.Prepend(CCFLAGS=["-Os"])

        if env["debug_symbols"]:
            env.Prepend(CCFLAGS=["-g2"])

    elif env["target"] == "release_debug":
        if env["optimize"] == "speed":  # optimize for speed (default)
            env.Prepend(CCFLAGS=["-O2"])
        elif env["optimize"] == "size":  # optimize for size
            env.Prepend(CCFLAGS=["-Os"])

        if env["debug_symbols"]:
            env.Prepend(CCFLAGS=["-g2"])

    elif env["target"] == "debug":
        env.Prepend(CCFLAGS=["-ggdb"])
        env.Prepend(CCFLAGS=["-g3"])
        env.Append(LINKFLAGS=["-rdynamic"])

    if env["debug_symbols"]:
        # Adding dwarf-4 explicitly makes stacktraces work with clang builds,
        # otherwise addr2line doesn't understand them
        env.Append(CCFLAGS=["-gdwarf-4"])

    ## Architecture

    is64 = sys.maxsize > 2**32
    if env["bits"] == "default":
        env["bits"] = "64" if is64 else "32"

    machines = {
        "riscv64": "rv64",
        "ppc64le": "ppc64",
        "ppc64": "ppc64",
        "ppcle": "ppc",
        "ppc": "ppc",
    }

    if env["arch"] == "" and platform.machine() in machines:
        env["arch"] = machines[platform.machine()]

    if env["arch"] == "rv64":
        # G = General-purpose extensions, C = Compression extension (very common).
        env.Append(CCFLAGS=["-march=rv64gc"])

    ## Compiler configuration

    if "CXX" in env and "clang" in os.path.basename(env["CXX"]):
        # Convenience check to enforce the use_llvm overrides when CXX is clang(++)
        env["use_llvm"] = True

    if env["use_llvm"]:
        if "clang++" not in os.path.basename(env["CXX"]):
            env["CC"] = "clang"
            env["CXX"] = "clang++"
        env.extra_suffix = ".llvm" + env.extra_suffix

    # Linker

    if env["linker"] != "default":
        print("Using linker program: " + env["linker"])
        if env["linker"] == "mold" and using_gcc(env):  # GCC < 12.1 doesn't support -fuse-ld=mold.
            cc_semver = tuple(get_compiler_version(env))
            if cc_semver < (12, 1):
                found_wrapper = False
                for path in ["/usr/libexec", "/usr/local/libexec", "/usr/lib", "/usr/local/lib"]:
                    if os.path.isfile(path + "/mold/ld"):
                        env.Append(LINKFLAGS=["-B" + path + "/mold"])
                        found_wrapper = True
                        break
                if not found_wrapper:
                    print("Couldn't locate mold installation path. Make sure it's installed in /usr or /usr/local.")
                    sys.exit(255)
            else:
                env.Append(LINKFLAGS=["-fuse-ld=mold"])
        else:
            env.Append(LINKFLAGS=["-fuse-ld=%s" % env["linker"]])

    if env['use_coverage']:
        env.Append(CCFLAGS=['-ftest-coverage', '-fprofile-arcs'])
        env.Append(LINKFLAGS=['-ftest-coverage', '-fprofile-arcs'])

    # Sanitizers
    if env["use_ubsan"] or env["use_asan"] or env["use_lsan"] or env["use_tsan"] or env["use_msan"]:
        env.extra_suffix += "s"

        if env["use_ubsan"]:
            env.Append(
                CCFLAGS=[
                    "-fsanitize=undefined,shift,shift-exponent,integer-divide-by-zero,unreachable,vla-bound,null,return,signed-integer-overflow,bounds,float-divide-by-zero,float-cast-overflow,nonnull-attribute,returns-nonnull-attribute,bool,enum,vptr,pointer-overflow,builtin"
                ]
            )

            if env["use_llvm"]:
                env.Append(
                    CCFLAGS=[
                        "-fsanitize=nullability-return,nullability-arg,function,nullability-assign,implicit-integer-sign-change,implicit-signed-integer-truncation,implicit-unsigned-integer-truncation"
                    ]
                )
            else:
                env.Append(CCFLAGS=["-fsanitize=bounds-strict"])
        env.Append(LINKFLAGS=["-fsanitize=undefined"])

        if env["use_asan"]:
            env.Append(CCFLAGS=["-fsanitize=address,pointer-subtract,pointer-compare"])
            env.Append(LINKFLAGS=["-fsanitize=address"])

        if env["use_lsan"]:
            env.Append(CCFLAGS=["-fsanitize=leak"])
            env.Append(LINKFLAGS=["-fsanitize=leak"])

        if env["use_tsan"]:
            env.Append(CCFLAGS=["-fsanitize=thread"])
            env.Append(LINKFLAGS=["-fsanitize=thread"])

        if env["use_msan"]:
            env.Append(CCFLAGS=["-fsanitize=memory"])
            env.Append(LINKFLAGS=["-fsanitize=memory"])

    # LTO

    if env["lto"] == "auto":  # Full LTO for production.
        env["lto"] = "full"

    if env["lto"] != "none":
        if env["lto"] == "thin":
            if not env["use_llvm"]:
                print("ThinLTO is only compatible with LLVM, use `use_llvm=yes` or `lto=full`.")
                sys.exit(255)
            env.Append(CCFLAGS=["-flto=thin"])
            env.Append(LINKFLAGS=["-flto=thin"])
        elif not env["use_llvm"] and env.GetOption("num_jobs") > 1:
            env.Append(CCFLAGS=["-flto"])
            env.Append(LINKFLAGS=["-flto=" + str(env.GetOption("num_jobs"))])
        else:
            env.Append(CCFLAGS=["-flto"])
            env.Append(LINKFLAGS=["-flto"])

        if not env["use_llvm"]:
            env["RANLIB"] = "gcc-ranlib"
            env["AR"] = "gcc-ar"

    env.Append(CCFLAGS=["-pipe"])

    # Check for gcc version >= 6 before adding -no-pie
    version = get_compiler_version(env) or [-1, -1]
    if using_gcc(env):
        if version[0] >= 6:
            env.Append(CCFLAGS=["-fpie"])
            env.Append(LINKFLAGS=["-no-pie"])
    # Do the same for clang should be fine with Clang 4 and higher
    if using_clang(env):
        if version[0] >= 4:
            env.Append(CCFLAGS=["-fpie"])
            env.Append(LINKFLAGS=["-no-pie"])

    env.Prepend(CPPPATH=["#platform/x11"])
    env.Append(CPPDEFINES=["X11_ENABLED", "UNIX_ENABLED", ("_FILE_OFFSET_BITS", 64)])

    env.Append(LIBS=["pthread"])

    env.Append(LIBS=["m"])

    if platform.system() == "Linux":
        env.Append(LIBS=["dl"])

    if not env["execinfo"] and platform.libc_ver()[0] != "glibc":
        # The default crash handler depends on glibc, so if the host uses
        # a different libc (BSD libc, musl), fall back to libexecinfo.
        print("Note: Using `execinfo=yes` for the crash handler as required on platforms where glibc is missing.")
        env["execinfo"] = True

    if env["execinfo"]:
        env.Append(LIBS=["execinfo"])

    ## Cross-compilation

    if is64 and env["bits"] == "32":
        env.Append(CCFLAGS=["-m32"])
        env.Append(LINKFLAGS=["-m32", "-L/usr/lib/i386-linux-gnu"])
    elif not is64 and env["bits"] == "64":
        env.Append(CCFLAGS=["-m64"])
        env.Append(LINKFLAGS=["-m64", "-L/usr/lib/i686-linux-gnu"])

    # Link those statically for portability
    if env["use_static_cpp"]:
        env.Append(LINKFLAGS=["-static-libgcc", "-static-libstdc++"])
        if env["use_llvm"] and platform.system() != "FreeBSD":
            #env["LINKCOM"] = env["LINKCOM"] + " -l:libatomic.a"
            env["LINKCOM"] = env["LINKCOM"] + " -latomic"

    else:
        if env["use_llvm"] and platform.system() != "FreeBSD":
            env.Append(LIBS=["atomic"])


    env.Append(LIBS=["stdc++"])
