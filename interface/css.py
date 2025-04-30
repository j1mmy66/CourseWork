css = """
:root {
    --primary: #00b894;
    --primary-hover: #00cec9;
    --bg: #2d3436;
    --card-bg: #2d3436;
    --text: #2d3436;
    --radius: 12px;
}

/* общий фон и шрифт */
body {
    background-color: var(--bg);
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: var(--text);
    margin: 0;
    padding: 0;
}

/* стили для кнопок */
.hover-button {
    background-color: var(--primary);
    color: white;
    border: none;
    border-radius: var(--radius);
    padding: 10px 20px;
    font-weight: bold;
    cursor: pointer;
    transition: background-color 0.3s ease;
    margin: 5px;
}
.hover-button:hover {
    background-color: var(--primary-hover);
}

/* стили для выпадающих списков */
.custom-dropdown .dropdown, 
.custom-dropdown .dropdown-text {
    border: 2px solid var(--primary);
    border-radius: var(--radius);
    padding: 8px 12px;
    background-color: var(--card-bg);
    color: var(--text);
    width: 200px;
    margin: 5px;
}

/* стили для табов */
.custom-tabs .gr-tabs {
    background-color: var(--card-bg);
    border-radius: var(--radius);
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.custom-tabs .gr-tab {
    color: var(--text);
    font-weight: bold;
    border-radius: var(--radius);
}
.custom-tabs .gr-tab[aria-selected="true"] {
    border-radius: var(--radius);
    border-bottom: 3px solid var(--primary);
    color: var(--primary);
}

/* «карточка» внутри таба */
.custom-card {
    background-color: var(--card-bg);
    border-radius: var(--radius);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    padding: 20px;
    margin-bottom: 20px;
}

/* заголовки секций */
.card-title h2 {
    color: var(--primary);
    margin-top: 0;
    margin-bottom: 15px;
}

/* стили для картинок-результатов */
.custom-image img {
    border-radius: var(--radius);
    border: 2px solid var(--primary);
    max-width: 100%;
    margin-top: 10px;
}
"""