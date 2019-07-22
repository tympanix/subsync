from setuptools import setup

exec(open('subsync/version.py').read())

setup(name='subsync',
      version=__version__,
      description='Synchronize your subtitles with machine learning',
      classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Multimedia :: Sound/Audio :: Speech',
      ],
      keywords='subtitle synchronize machine learning',
      platforms=["Independent"],
      scripts=['subsync/bin/subsync'],
      include_package_data=True,
      url='https://github.com/tympanix/subsync',
      author='tympanix',
      author_email='tympanix@gmail.com',
      license='MIT',
      packages=['subsync'],
      install_requires=[
          'tensorflow>=1.0.0',
          'numpy',
          'matplotlib',
          'librosa',
          'h5py>=2.9.0',
          'pysrt',
      ],
      zip_safe=False)
