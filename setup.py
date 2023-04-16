# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import subprocess
from pathlib import Path

import setuptools
import setuptools.command.build_ext


class make_ext(setuptools.command.build_ext.build_ext):  # type:ignore[misc]
    def build_extension(self, ext: setuptools.Extension) -> None:
        if ext.name == "besskge_custom_ops":
            filename = Path(self.build_lib) / self.get_ext_filename(
                self.get_ext_fullname(ext.name)
            )
            objdir = filename.with_suffix("")
            subprocess.check_call(
                [
                    "make",
                    f"OUT={filename}",
                    f"OBJDIR={objdir}",
                ]
            )
        else:
            super().build_extension(ext)


setuptools.setup(
    name="besskge",
    version="0.1",
    install_requires=Path("requirements.txt").read_text().rstrip("\n").split("\n"),
    ext_modules=[
        setuptools.Extension(
            "besskge_custom_ops",
            list(map(str, Path("besskge/custom_ops").glob("*.[ch]pp"))),
        )
    ],
    package_data={"besskge": ["custom_ops/*_codelet.cpp"]},
    cmdclass=dict(build_ext=make_ext),
)
