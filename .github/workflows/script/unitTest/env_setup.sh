pip list

inc=$(pip list | grep -c 'neural-compressor') || true # Prevent from exiting when 'inc' not found
if [ ${inc} != 0 ]; then
    pip uninstall neural-compressor -y
    pip list
fi

echo "Install neural_compressor binary..."

n=0
until [ "$n" -ge 5 ]; do
    git clone https://github.com/intel/neural-compressor.git /neural-compressor
    cd /neural-compressor
    pip install -r requirements.txt
    python setup.py install && break
    n=$((n + 1))
    sleep 5
done

# Install test requirements
cd $1 || exit 1
if [ -f "requirements.txt" ]; then
    python -m pip install --default-timeout=100 -r requirements.txt
    pip list
else
    echo "Not found requirements.txt file."
fi

pip install coverage
pip install pytest
