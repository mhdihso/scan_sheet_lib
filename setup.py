from setuptools import setup

setup(
    name='open-cy paper detect',
    version='0.1.2',    
    description='open-cy paper detect for exams',
    url='https://gitlab.com/behraad/openCvCreator',
    author='Behrad Jafari',
    author_email='be.jafari@madtalk.ir',
    license='MIT',
    install_requires=['numpy' , 'opencv-python' , 'scikit-image' , 'imutils' , 'matplotlib' ],
    packages=['noyan_opencv'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
