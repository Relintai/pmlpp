#!/usr/bin/env python

EnsureSConsVersion(0, 98, 1)

# System
import atexit
import glob
import os
import pickle
import sys
import time
from collections import OrderedDict

# Local
import methods
from platform_methods import run_in_subprocess

# scan possible build platforms

platform_list = []  # list of platforms
platform_opts = {}  # options for each platform
platform_flags = {}  # flags for each platform

active_platforms = []
active_platform_ids = []
platform_exporters = []
platform_apis = []

time_at_start = time.time()

def add_mlpp(env):
    env.modules_sources = []
    module_env = env.Clone()

    module_env.pmlpp_build_tests = True

    if ARGUMENTS.get('pmlpp_build_tests', 'yes') == 'no':
        module_env.pmlpp_build_tests = False

    sources = [
        "core/mlpp_vector.cpp",
        "core/mlpp_matrix.cpp",
        "core/mlpp_tensor3.cpp",

        "core/activation.cpp",
        "core/convolutions.cpp",
        "core/cost.cpp",
        "core/data.cpp",
        "core/lin_alg.cpp",
        "core/numerical_analysis.cpp",
        "core/transforms.cpp",
        "core/stat.cpp",
        "core/utilities.cpp",
        "core/hypothesis_testing.cpp",
        "core/reg.cpp",

        "ann/ann.cpp",
        "auto_encoder/auto_encoder.cpp",
        "bernoulli_nb/bernoulli_nb.cpp",
        "c_log_log_reg/c_log_log_reg.cpp",
        "dual_svc/dual_svc.cpp",
        "exp_reg/exp_reg.cpp",
        "gan/gan.cpp",
        "gaussian_nb/gaussian_nb.cpp",
        "gauss_markov_checker/gauss_markov_checker.cpp",
        "hidden_layer/hidden_layer.cpp",
        "kmeans/kmeans.cpp",
        "knn/knn.cpp",
        "lin_reg/lin_reg.cpp",
        "log_reg/log_reg.cpp",
        "mann/mann.cpp",
        "mlp/mlp.cpp",
        "multinomial_nb/multinomial_nb.cpp",
        "multi_output_layer/multi_output_layer.cpp",
        "outlier_finder/outlier_finder.cpp",
        "output_layer/output_layer.cpp",
        "pca/pca.cpp",
        "probit_reg/probit_reg.cpp",
        "softmax_net/softmax_net.cpp",
        "softmax_reg/softmax_reg.cpp",
        "svc/svc.cpp",
        "tanh_reg/tanh_reg.cpp",
        "uni_lin_reg/uni_lin_reg.cpp",
        "wgan/wgan.cpp",
    ]

    if module_env.pmlpp_build_tests:
        module_env.Prepend(CPPDEFINES=["TESTS_ENABLED"])

        sources += [
            "test/mlpp_tests.cpp",
            "test/mlpp_matrix_tests.cpp",
        ]


    if ARGUMENTS.get('pmlpp_shared', 'no') == 'yes':
        # Shared lib compilation
        module_env.Append(CCFLAGS=['-fPIC'])
        module_env['LIBS'] = []
        shared_lib = module_env.SharedLibrary(target='#bin/pmlpp', source=sources)
        shared_lib_shim = shared_lib[0].name.rsplit('.', 1)[0]
        env.Append(LIBS=[shared_lib_shim])
        env.Append(LIBPATH=['#bin'])
    else:
        # Static compilation
        module_env.add_source_files(env.modules_sources, sources)
        lib = env.add_library("mlpp", env.modules_sources)
        env.Prepend(LIBS=[lib])

for x in sorted(glob.glob("platform/*")):
    if not os.path.isdir(x) or not os.path.exists(x + "/detect.py"):
        continue
    tmppath = "./" + x

    sys.path.insert(0, tmppath)
    import detect

    if detect.is_active():
        active_platforms.append(detect.get_name())
        active_platform_ids.append(x)
    if detect.can_build():
        x = x.replace("platform/", "")  # rest of world
        x = x.replace("platform\\", "")  # win32
        platform_list += [x]
        platform_opts[x] = detect.get_opts()
        platform_flags[x] = detect.get_flags()
    sys.path.remove(tmppath)
    sys.modules.pop("detect")

custom_tools = ["default"]

platform_arg = ARGUMENTS.get("platform", ARGUMENTS.get("p", False))

if platform_arg == "android":
    custom_tools = ["clang", "clang++", "as", "ar", "link"]
elif platform_arg == "javascript":
    # Use generic POSIX build toolchain for Emscripten.
    custom_tools = ["cc", "c++", "ar", "link", "textfile", "zip"]
elif os.name == "nt" and methods.get_cmdline_bool("use_mingw", False):
    custom_tools = ["mingw"]

# We let SCons build its default ENV as it includes OS-specific things which we don't
# want to have to pull in manually.
# Then we prepend PATH to make it take precedence, while preserving SCons' own entries.
env_base = Environment(tools=custom_tools)
env_base.PrependENVPath("PATH", os.getenv("PATH"))
env_base.PrependENVPath("PKG_CONFIG_PATH", os.getenv("PKG_CONFIG_PATH"))
if "TERM" in os.environ:  # Used for colored output.
    env_base["ENV"]["TERM"] = os.environ["TERM"]

env_base.disabled_modules = []
env_base.use_ptrcall = False
env_base.module_version_string = ""
env_base.msvc = False

env_base.__class__.disable_module = methods.disable_module

env_base.__class__.add_module_version_string = methods.add_module_version_string

env_base.__class__.add_source_files = methods.add_source_files
env_base.__class__.use_windows_spawn_fix = methods.use_windows_spawn_fix
env_base.__class__.split_lib = methods.split_lib

env_base.__class__.add_shared_library = methods.add_shared_library
env_base.__class__.add_library = methods.add_library
env_base.__class__.add_program = methods.add_program
env_base.__class__.CommandNoCache = methods.CommandNoCache
env_base.__class__.disable_warnings = methods.disable_warnings
env_base.__class__.force_optimization_on_debug = methods.force_optimization_on_debug
env_base.__class__.module_add_dependencies = methods.module_add_dependencies
env_base.__class__.module_check_dependencies = methods.module_check_dependencies

env_base["x86_libtheora_opt_gcc"] = False
env_base["x86_libtheora_opt_vc"] = False

# avoid issues when building with different versions of python out of the same directory
env_base.SConsignFile(".sconsign{0}.dblite".format(pickle.HIGHEST_PROTOCOL))

# Build options

customs = ["custom.py"]

profile = ARGUMENTS.get("profile", "")
if profile:
    if os.path.isfile(profile):
        customs.append(profile)
    elif os.path.isfile(profile + ".py"):
        customs.append(profile + ".py")

opts = Variables(customs, ARGUMENTS)

# Target build options
opts.Add("p", "Platform (alias for 'platform')", "")
opts.Add("platform", "Target platform (%s)" % ("|".join(platform_list),), "")
opts.Add(EnumVariable("target", "Compilation target", "debug", ("debug", "release_debug", "release")))
opts.Add("arch", "Platform-dependent architecture (arm/arm64/x86/x64/mips/...)", "")
opts.Add(EnumVariable("bits", "Target platform bits", "default", ("default", "32", "64")))
opts.Add(EnumVariable("optimize", "Optimization type", "speed", ("speed", "size", "none")))
opts.Add(BoolVariable("production", "Set defaults to build Pandemonium for use in production", False))
opts.Add(EnumVariable("lto", "Link-time optimization (production builds)", "none", ("none", "auto", "thin", "full")))
opts.Add(BoolVariable("use_rtti", "Use RTTI", False))

# Components
opts.Add(BoolVariable("deprecated", "Enable deprecated features", True))
opts.Add(BoolVariable("minizip", "Enable ZIP archive support using minizip", True))
opts.Add(BoolVariable("xaudio2", "Enable the XAudio2 audio driver", False))
opts.Add(BoolVariable("disable_exceptions", "Force disabling exception handling code", True))

# Advanced options
opts.Add(BoolVariable("dev", "If yes, alias for verbose=yes warnings=extra werror=yes", False))
opts.Add(BoolVariable("fast_unsafe", "Enable unsafe options for faster rebuilds", False))
opts.Add(BoolVariable("compiledb", "Generate compilation DB (`compile_commands.json`) for external tools", False))
opts.Add(BoolVariable("verbose", "Enable verbose output for the compilation", False))
opts.Add(BoolVariable("progress", "Show a progress indicator during compilation", True))
opts.Add(EnumVariable("warnings", "Level of compilation warnings", "all", ("extra", "all", "moderate", "no")))
opts.Add(BoolVariable("werror", "Treat compiler warnings as errors", False))
opts.Add("extra_suffix", "Custom extra suffix added to the base filename of all generated binary files", "")
opts.Add(BoolVariable("vsproj", "Generate a Visual Studio solution", False))
opts.Add(
    BoolVariable(
        "split_libmodules",
        "Split intermediate libmodules.a in smaller chunks to prevent exceeding linker command line size (forced to True when using MinGW)",
        False,
    )
)
opts.Add(BoolVariable("use_precise_math_checks", "Math checks use very precise epsilon (debug option)", False))
opts.Add("main_scsub_path", "The SCSub's path that has the main method.", "platform/main/SCsub")

# Compilation environment setup
opts.Add("CXX", "C++ compiler")
opts.Add("CC", "C compiler")
opts.Add("LINK", "Linker")
opts.Add("CCFLAGS", "Custom flags for both the C and C++ compilers")
opts.Add("CFLAGS", "Custom flags for the C compiler")
opts.Add("CXXFLAGS", "Custom flags for the C++ compiler")
opts.Add("LINKFLAGS", "Custom flags for the linker")

# Update the environment to have all above options defined
# in following code (especially platform and custom_modules).
opts.Update(env_base)

# Platform selection: validate input, and add options.

selected_platform = ""

if env_base["platform"] != "":
    selected_platform = env_base["platform"]
elif env_base["p"] != "":
    selected_platform = env_base["p"]
else:
    # Missing `platform` argument, try to detect platform automatically
    if (
        sys.platform.startswith("linux")
        or sys.platform.startswith("dragonfly")
        or sys.platform.startswith("freebsd")
        or sys.platform.startswith("netbsd")
        or sys.platform.startswith("openbsd")
    ):
        selected_platform = "x11"
    elif sys.platform == "darwin":
        selected_platform = "osx"
    elif sys.platform == "win32":
        selected_platform = "windows"
    else:
        print("Could not detect platform automatically. Supported platforms:")
        for x in platform_list:
            print("\t" + x)
        print("\nPlease run SCons again and select a valid platform: platform=<string>")

    if selected_platform != "":
        print("Automatically detected platform: " + selected_platform)

if selected_platform == "macos":
    # Alias for forward compatibility.
    print('Platform "macos" is still called "osx" in Godot 3.x. Building for platform "osx".')
    selected_platform = "osx"

if selected_platform == "ios":
    # Alias for forward compatibility.
    print('Platform "ios" is still called "iphone" in Godot 3.x. Building for platform "iphone".')
    selected_platform = "iphone"

if selected_platform in ["linux", "bsd", "linuxbsd"]:
    if selected_platform == "linuxbsd":
        # Alias for forward compatibility.
        print('Platform "linuxbsd" is still called "x11" in Pandemonium 3.x. Building for platform "x11".')
    # Alias for convenience.
    selected_platform = "x11"

# Make sure to update this to the found, valid platform as it's used through the buildsystem as the reference.
# It should always be re-set after calling `opts.Update()` otherwise it uses the original input value.
env_base["platform"] = selected_platform

# Add platform-specific options.
if selected_platform in platform_opts:
    for opt in platform_opts[selected_platform]:
        opts.Add(opt)

# Update the environment to take platform-specific options into account.
opts.Update(env_base)
env_base["platform"] = selected_platform  # Must always be re-set after calling opts.Update().

# Update the environment again after all the module options are added.
opts.Update(env_base)
env_base["platform"] = selected_platform  # Must always be re-set after calling opts.Update().
Help(opts.GenerateHelpText(env_base))

# add default include paths

env_base.Prepend(CPPPATH=["#"])
env_base.Prepend(CPPPATH=["#platform"])

# USE SFWL
env_base.Prepend(CPPDEFINES=["USING_SFW"])

# configure ENV for platform
env_base.platform_exporters = platform_exporters
env_base.platform_apis = platform_apis

# Build type defines - more platform-specific ones can be in detect.py.
if env_base["target"] == "release_debug" or env_base["target"] == "debug":
    # DEBUG_ENABLED enables debugging *features* and debug-only code, which is intended
    # to give *users* extra debugging information for their game development.
    env_base.Append(CPPDEFINES=["DEBUG_ENABLED"])

if env_base["target"] == "debug":
    # DEV_ENABLED enables *engine developer* code which should only be compiled for those
    # working on the engine itself.
    env_base.Append(CPPDEFINES=["DEV_ENABLED"])
else:
    # Disable assert() for production targets (only used in thirdparty code).
    env_base.Append(CPPDEFINES=["NDEBUG"])

# SCons speed optimization controlled by the `fast_unsafe` option, which provide
# more than 10 s speed up for incremental rebuilds.
# Unsafe as they reduce the certainty of rebuilding all changed files, so it's
# enabled by default for `debug` builds, and can be overridden from command line.
# Ref: https://github.com/SCons/scons/wiki/GoFastButton
if methods.get_cmdline_bool("fast_unsafe", env_base["target"] == "debug"):
    # Renamed to `content-timestamp` in SCons >= 4.2, keeping MD5 for compat.
    env_base.Decider("MD5-timestamp")
    env_base.SetOption("implicit_cache", 1)
    env_base.SetOption("max_drift", 60)

if env_base["use_precise_math_checks"]:
    env_base.Append(CPPDEFINES=["PRECISE_MATH_CHECKS"])

if not env_base["deprecated"]:
    env_base.Append(CPPDEFINES=["DISABLE_DEPRECATED"])

#env_base.Append(LIBS=["stdc++"])

if selected_platform in platform_list:
    tmppath = "./platform/" + selected_platform
    sys.path.insert(0, tmppath)
    import detect

    env = env_base.Clone()

    env.extra_suffix = ""

    if env["extra_suffix"] != "":
        env.extra_suffix += "." + env["extra_suffix"]

    # Environment flags
    CCFLAGS = env.get("CCFLAGS", "")
    env["CCFLAGS"] = ""
    env.Append(CCFLAGS=str(CCFLAGS).split())

    CFLAGS = env.get("CFLAGS", "")
    env["CFLAGS"] = ""
    env.Append(CFLAGS=str(CFLAGS).split())

    CXXFLAGS = env.get("CXXFLAGS", "")
    env["CXXFLAGS"] = ""
    env.Append(CXXFLAGS=str(CXXFLAGS).split())

    LINKFLAGS = env.get("LINKFLAGS", "")
    env["LINKFLAGS"] = ""
    env.Append(LINKFLAGS=str(LINKFLAGS).split())

    # Platform specific flags.
    # These can sometimes override default options.
    flag_list = platform_flags[selected_platform]
    for f in flag_list:
        if not (f[0] in ARGUMENTS):  # allow command line to override platform flags
            env[f[0]] = f[1]

    # 'dev' and 'production' are aliases to set default options if they haven't been
    # set manually by the user.
    # These need to be checked *after* platform specific flags so that different
    # default values can be set (e.g. to keep LTO off for `production` on some platforms).
    if env["dev"]:
        env["verbose"] = methods.get_cmdline_bool("verbose", True)
        env["warnings"] = ARGUMENTS.get("warnings", "extra")
        env["werror"] = methods.get_cmdline_bool("werror", True)
    if env["production"]:
        env["use_static_cpp"] = methods.get_cmdline_bool("use_static_cpp", True)
        env["debug_symbols"] = methods.get_cmdline_bool("debug_symbols", False)
        # LTO "auto" means we handle the preferred option in each platform detect.py.
        env["lto"] = ARGUMENTS.get("lto", "auto")

    # Must happen after the flags' definition, as configure is when most flags
    # are actually handled to change compile options, etc.
    detect.configure(env)

    # Needs to happen after configure to handle "auto".
    if env["lto"] != "none":
        print("Using LTO: " + env["lto"])

    # Set our C and C++ standard requirements.
    # Prepending to make it possible to override
    # This needs to come after `configure`, otherwise we don't have env.msvc.
    if not env.msvc:
        # Specifying GNU extensions support explicitly, which are supported by
        # both GCC and Clang. This mirrors GCC and Clang's current default
        # compile flags if no -std is specified.
        env.Prepend(CFLAGS=["-std=gnu11"])
        env.Prepend(CXXFLAGS=["-std=gnu++14"])
    else:
        # MSVC doesn't have clear C standard support, /std only covers C++.
        # We apply it to CCFLAGS (both C and C++ code) in case it impacts C features.
        env.Prepend(CCFLAGS=["/std:c++14"])

    if env_base["use_rtti"]:
        if not env.msvc:
            env_base.Prepend(CXXFLAGS=["-frtti"])
            env.Prepend(CXXFLAGS=["-frtti"])
        else:
            env_base.Prepend(CXXFLAGS=["/GR"])
            env.Prepend(CXXFLAGS=["/GR"])
    else:
        if not env.msvc:
            env_base.Prepend(CXXFLAGS=["-fno-rtti"])
            env.Prepend(CXXFLAGS=["-fno-rtti"])
        else:
            env_base.Prepend(CXXFLAGS=["/GR-"])
            env.Prepend(CXXFLAGS=["/GR-"])
        
        # Don't use dynamic_cast, necessary with no-rtti.
        env_base.Prepend(CPPDEFINES=["NO_SAFE_CAST"])
        env.Prepend(CPPDEFINES=["NO_SAFE_CAST"])

    # Handle renamed options.
    if "use_lto" in ARGUMENTS or "use_thinlto" in ARGUMENTS:
        print("Error: The `use_lto` and `use_thinlto` boolean options have been unified to `lto=<none|thin|full>`.")
        print("       Please adjust your scripts accordingly.")
        Exit(255)
    if "use_lld" in ARGUMENTS:
        print("Error: The `use_lld` boolean option has been replaced by `linker=<default|bfd|gold|lld|mold>`.")
        print("       Please adjust your scripts accordingly.")
        Exit(255)

    # Disable exception handling. Godot doesn't use exceptions anywhere, and this
    # saves around 20% of binary size and very significant build time (GH-80513).
    if env["disable_exceptions"]:
        if env.msvc:
            env.Append(CPPDEFINES=[("_HAS_EXCEPTIONS", 0)])
        else:
            env.Append(CCFLAGS=["-fno-exceptions"])
    elif env.msvc:
        env.Append(CCFLAGS=["/EHsc"])

    # Configure compiler warnings
    if env.msvc:  # MSVC
        # Truncations, narrowing conversions, signed/unsigned comparisons...
        disable_nonessential_warnings = ["/wd4267", "/wd4244", "/wd4305", "/wd4018", "/wd4800"]
        if env["warnings"] == "extra":
            env.Append(CCFLAGS=["/Wall"])  # Implies /W4
        elif env["warnings"] == "all":
            env.Append(CCFLAGS=["/W3"] + disable_nonessential_warnings)
        elif env["warnings"] == "moderate":
            env.Append(CCFLAGS=["/W2"] + disable_nonessential_warnings)
        else:  # 'no'
            env.Append(CCFLAGS=["/w"])

        if env["werror"]:
            env.Append(CCFLAGS=["/WX"])
            env.Append(LINKFLAGS=["/WX"])
    else:  # GCC, Clang
        version = methods.get_compiler_version(env) or [-1, -1]

        common_warnings = []

        if methods.using_gcc(env):
            common_warnings += ["-Wno-misleading-indentation"]
            if version[0] >= 7:
                common_warnings += ["-Wshadow-local"]
        elif methods.using_clang(env) or methods.using_emcc(env):
            # We often implement `operator<` for structs of pointers as a requirement
            # for putting them in `Set` or `Map`. We don't mind about unreliable ordering.
            common_warnings += ["-Wno-ordered-compare-function-pointers"]

        if env["warnings"] == "extra":
            # Note: enable -Wimplicit-fallthrough for Clang (already part of -Wextra for GCC)
            # once we switch to C++11 or later (necessary for our FALLTHROUGH macro).
            env.Append(CCFLAGS=["-Wall", "-Wextra", "-Wwrite-strings", "-Wno-unused-parameter"] + common_warnings)
            env.Append(CXXFLAGS=["-Wctor-dtor-privacy", "-Wnon-virtual-dtor"])
            if methods.using_gcc(env):
                env.Append(
                    CCFLAGS=[
                        "-Walloc-zero",
                        "-Wduplicated-branches",
                        "-Wduplicated-cond",
                        "-Wstringop-overflow=4",
                        "-Wlogical-op",
                    ]
                )
                env.Append(CXXFLAGS=["-Wnoexcept", "-Wplacement-new=1"])
                if version[0] >= 9:
                    env.Append(CCFLAGS=["-Wattribute-alias=2"])
        elif env["warnings"] == "all":
            env.Append(CCFLAGS=["-Wall"] + common_warnings)
        elif env["warnings"] == "moderate":
            env.Append(CCFLAGS=["-Wall", "-Wno-unused"] + common_warnings)
        else:  # 'no'
            env.Append(CCFLAGS=["-w"])

        if env["werror"]:
            env.Append(CCFLAGS=["-Werror"])
            if methods.using_gcc(env) and version[0] >= 12:  # False positives in our error macros, see GH-58747.
                env.Append(CCFLAGS=["-Wno-error=return-type"])

    if hasattr(detect, "get_program_suffix"):
        suffix = "." + detect.get_program_suffix()
    else:
        suffix = "." + selected_platform

    if env["target"] == "release":
        suffix += ".opt"
    elif env["target"] == "release_debug":
        suffix += ".opt.debug"
    else:
        print(
            "Note: Building a debug binary (which will run slowly). Use `target=release` to build an optimized release binary."
        )
        suffix += ".debug"

    if env["arch"] != "":
        suffix += "." + env["arch"]
    elif env["bits"] == "32":
        suffix += ".32"
    elif env["bits"] == "64":
        suffix += ".64"

    suffix += env.extra_suffix

    sys.path.remove(tmppath)
    sys.modules.pop("detect")

    modules_enabled = OrderedDict()
    env.module_dependencies = {}
    env.module_icons_paths = []
    env.module_license_files = []
    env.doc_class_path = {}

    env.module_list = modules_enabled
    methods.sort_module_list(env)

    env["PROGSUFFIX"] = suffix + env.module_version_string + env["PROGSUFFIX"]
    env["OBJSUFFIX"] = suffix + env["OBJSUFFIX"]
    # (SH)LIBSUFFIX will be used for our own built libraries
    # LIBSUFFIXES contains LIBSUFFIX and SHLIBSUFFIX by default,
    # so we need to append the default suffixes to keep the ability
    # to link against thirdparty libraries (.a, .so, .lib, etc.).
    if os.name == "nt":
        # On Windows, only static libraries and import libraries can be
        # statically linked - both using .lib extension
        env["LIBSUFFIXES"] += [env["LIBSUFFIX"]]
    else:
        env["LIBSUFFIXES"] += [env["LIBSUFFIX"], env["SHLIBSUFFIX"]]
    env["LIBSUFFIX"] = suffix + env["LIBSUFFIX"]
    env["SHLIBSUFFIX"] = suffix + env["SHLIBSUFFIX"]

    if env.use_ptrcall:
        env.Append(CPPDEFINES=["PTRCALL_ENABLED"])

    if not env["verbose"]:
        methods.no_verbose(sys, env)

    scons_cache_path = os.environ.get("SCONS_CACHE")
    if scons_cache_path != None:
        CacheDir(scons_cache_path)
        print("Scons cache enabled... (path: '" + scons_cache_path + "')")

    if env["vsproj"]:
        env.vs_incs = []
        env.vs_srcs = []

    if env["compiledb"]:
        # Generating the compilation DB (`compile_commands.json`) requires SCons 4.0.0 or later.
        from SCons import __version__ as scons_raw_version

        scons_ver = env._get_major_minor_revision(scons_raw_version)

        if scons_ver < (4, 0, 0):
            print("The `compiledb=yes` option requires SCons 4.0 or later, but your version is %s." % scons_raw_version)
            Exit(255)

        env.Tool("compilation_db")
        env.Alias("compiledb", env.CompilationDatabase())

    Export("env")

    # build subdirs, the build order is dependent on link order.
    
    SConscript("platform/SCsub")

    SConscript("platform/" + selected_platform + "/SCsub")  # build selected platform

    add_mlpp(env)

    SConscript(env["main_scsub_path"])

    # Microsoft Visual Studio Project Generation
    if env["vsproj"]:
        if os.name != "nt":
            print("Error: The `vsproj` option is only usable on Windows with Visual Studio.")
            Exit(255)
        env["CPPPATH"] = [Dir(path) for path in env["CPPPATH"]]
        methods.generate_vs_project(env, GetOption("num_jobs"))
        methods.generate_cpp_hint_file("cpp.hint")

    # Check for the existence of headers
    conf = Configure(env)
    if "check_c_headers" in env:
        for header in env["check_c_headers"]:
            if conf.CheckCHeader(header[0]):
                env.AppendUnique(CPPDEFINES=[header[1]])

elif selected_platform != "":
    if selected_platform == "list":
        print("The following platforms are available:\n")
    else:
        print('Invalid target platform "' + selected_platform + '".')
        print("The following platforms were detected:\n")

    for x in platform_list:
        print("\t" + x)

    print("\nPlease run SCons again and select a valid platform: platform=<string>")

    if selected_platform == "list":
        # Exit early to suppress the rest of the built-in SCons messages
        sys.exit(0)
    else:
        sys.exit(255)

# The following only makes sense when the 'env' is defined, and assumes it is.
if "env" in locals():
    # FIXME: This method mixes both cosmetic progress stuff and cache handling...
    methods.show_progress(env)
    # TODO: replace this with `env.Dump(format="json")`
    # once we start requiring SCons 4.0 as min version.
    methods.dump(env)


def print_elapsed_time():
    elapsed_time_sec = round(time.time() - time_at_start, 3)
    time_ms = round((elapsed_time_sec % 1) * 1000)
    print("[Time elapsed: {}.{:03}]".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time_sec)), time_ms))


atexit.register(print_elapsed_time)

