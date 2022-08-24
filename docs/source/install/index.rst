Installing AITemplate
=====================


Pre-Install inside Meta
--------------------
- Install Enabling ttls_fwdproxy

.. code-block:: bash

    sudo feature install ttls_fwdproxy
    systemctl --user start ttls_fwdproxy_loader.service
    echo "alias with-proxy=\"HTTPS_PROXY=http://fwdproxy:8080 HTTP_PROXY=http://fwdproxy:8080 FTP_PROXY=http://fwdproxy:8080 https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 ftp_proxy=http://fwdproxy:8080 http_no_proxy='\''*.facebook.com|*.tfbnw.net|*.fb.com'\'\"" >> ~/.bash_profile
    source ~/.bash_profil

- Install Python

.. code-block:: bash

    sudo dnf install python3.8


Standard Steps inside Meta
------------------------------------------
- Install required dependency

.. code-block:: bash

    with-proxy sudo pip3.8 install numpy
    with-proxy sudo pip3.8 install jinja2

- Install AITemplate

.. code-block:: bash

    with-proxy sudo python3.8 python/setup.py develop


- Install dependency for documents

.. code-block:: bash

    with-proxy sudo pip3.8 install autodocsumm
    with-proxy sudo pip3.8 install sphinx_rtd_theme
    with-proxy sudo pip3.8 install sphinx_gallery
    with-proxy sudo pip3.8 install sphinxcontrib-inlinesyntaxhighlight
    with-proxy sudo pip3.8 install sphinx_toolbox

- Install dependency for running unit tests

.. code-block:: bash

    with-proxy sudo pip3.8 install pytest
    with-proxy sudo pip3.8 install parameterized
    with-proxy sudo pip3.8 install ufmt
    with-proxy sudo pip3.8 install pylint==2.13.9
    # For CUDA
    with-proxy sudo pip3.8 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    # For ROCM
    with-proxy sudo install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm4.5.2


Standard Steps outside Meta
-----------------------------

When installing outside of Meta,
just replace `with-proxy sudo pip3.8` with standard `pip`
then it is good to go. We are going to provide Docker images in the future.
