vars = {
  # Pull in chromium build files and tools for multi-platform build support
  # of the test harness and the skia library.
  'chromium_git': 'https://chromium.googlesource.com/chromium/src',

  # Pull in skia source for image decode.
  'skia_git': 'https://skia.googlesource.com/skia',

  'mlpth_root': 'src',
}

deps = {
  '{mlpth_root}/build': {
    'url': '{chromium_git}/build@e3ed5e43c305b353b49e08ac69e7f4d1c2d88ad2'
  },
  '{mlpth_root}/buildtools': {
    'url': '{chromium_git}/buildtools@106e9fce3799633f42b45ca8bbe9e84e1e235603'
  },
  '{mlpth_root}/tools/clang': {
    'url': '{chromium_git}/tools/clang.git@3114fbc11f9644c54dd0a4cdbfa867bac50ff983',
  },
  '{mlpth_root}/third_party/cpython': {
    'url': 'https://github.com/python/cpython.git@3.7',
  },
  '{mlpth_root}/third_party/pybind': {
    'url': 'https://github.com/pybind/pybind11.git@v2.2',
  },
  # Pull the skia source in only for the headers at the moment.
  # We might compile as part of this project in the future.
  '{mlpth_root}/third_party/skia': {
    'url': '{skia_git}@7a1e00421d97a1f4f171dd2a8f817f9b07c9cdfd'
  },
}

recursedeps = [
  '{mlpth_root}/buildtools',
  '{mlpth_root}/third_party/skia',
]

#gclient_gn_args_from = 'src/buildtools'

hooks = [
  # Pull clang-format binaries using checked-in hashes.
  {
    'name': 'clang_format_win',
    'pattern': '.',
    'condition': 'host_os == "win"',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--platform=win32',
                '--no_auth',
                '--bucket', 'chromium-clang-format',
                '-s', '{mlpth_root}/buildtools/win/clang-format.exe.sha1',
    ],
  },
  {
    'name': 'clang_format_mac',
    'pattern': '.',
    'condition': 'host_os == "mac"',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--platform=darwin',
                '--no_auth',
                '--bucket', 'chromium-clang-format',
                '-s', '{mlpth_root}/buildtools/mac/clang-format.sha1',
    ],
  },
  {
    'name': 'clang_format_linux',
    'pattern': '.',
    'condition': 'host_os == "linux"',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--platform=linux*',
                '--no_auth',
                '--bucket', 'chromium-clang-format',
                '-s', '{mlpth_root}/buildtools/linux64/clang-format.sha1',
    ],
  },

  # Pull GN using checked-in hashes.
  {
    'name': 'gn_win',
    'pattern': '.',
    'condition': 'host_os == "win"',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--platform=win32',
                '--no_auth',
                '--bucket', 'chromium-gn',
                '-s', '{mlpth_root}/buildtools/win/gn.exe.sha1',
    ],
  },
  {
    'name': 'gn_mac',
    'pattern': '.',
    'condition': 'host_os == "mac"',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--platform=darwin',
                '--no_auth',
                '--bucket', 'chromium-gn',
                '-s', '{mlpth_root}/buildtools/mac/gn.sha1',
    ],
  },
  {
    'name': 'gn_linux',
    'pattern': '.',
    'condition': 'host_os == "linux"',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--platform=linux*',
                '--no_auth',
                '--bucket', 'chromium-gn',
                '-s', '{mlpth_root}/buildtools/linux64/gn.sha1',
    ],
  },

  # Pull sysroots.
  {
    'name': 'sysroot_arm',
    'pattern': '.',
    'condition': '(checkout_linux and checkout_arm)',
    'action': ['python', '{mlpth_root}/build/linux/sysroot_scripts/install-sysroot.py',
               '--arch=arm'],
  },
  {
    'name': 'sysroot_arm64',
    'pattern': '.',
    'condition': '(checkout_linux and checkout_arm64)',
    'action': ['python', '{mlpth_root}/build/linux/sysroot_scripts/install-sysroot.py',
               '--arch=arm64'],
  },
  {
    'name': 'sysroot_x86',
    'pattern': '.',
    'condition': '(checkout_linux and (checkout_x86 or checkout_x64))',
    'action': ['python', '{mlpth_root}/build/linux/sysroot_scripts/install-sysroot.py',
               '--arch=x86'],
  },
  {
    'name': 'sysroot_mips',
    'pattern': '.',
    'condition': '(checkout_linux and checkout_mips)',
    'action': ['python', '{mlpth_root}/build/linux/sysroot_scripts/install-sysroot.py',
               '--arch=mips'],
  },
  {
    'name': 'sysroot_x64',
    'pattern': '.',
    'condition': 'checkout_linux and checkout_x64',
    'action': ['python', '{mlpth_root}/build/linux/sysroot_scripts/install-sysroot.py',
               '--arch=x64'],
  },
  {
    # Update the Windows toolchain if necessary.
    'name': 'win_toolchain',
    'pattern': '.',
    'condition': 'checkout_win',
    'action': ['python', '{mlpth_root}/build/vs_toolchain.py', 'update'],
  },
  {
    'name': 'fuchsia_sdk',
    'pattern': '.',
    'condition': 'checkout_fuchsia',
    'action': [
      'python',
      '{mlpth_root}/build/fuchsia/update_sdk.py',
    ],
  },
  {
    # Note: On Win, this should run after win_toolchain, as it may use it.
    'name': 'clang',
    'pattern': '.',
    # clang not supported on aix
    'condition': 'host_os != "aix"',
    'action': ['python', '{mlpth_root}//tools/clang/scripts/update.py'],
  },
]
