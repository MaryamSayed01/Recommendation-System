mkdir -p ~/.streamlit/

echo "\
[general]\n\
email=\"your-email@domain.com\"\n\
"> ~/ .streamlit/credentials.toml

echo"\
[server]\n\
headless=true\n\
enableCORS=false\n\
<<<<<<< HEAD:setup.sh.txt
port= $PORT\n\
"> ~/ .streamlit/config.toml
=======
\n\
"> ~/ .streamlit/config.toml
>>>>>>> edceea0e14697133566b521feb6879d76872f08f:setup.sh
