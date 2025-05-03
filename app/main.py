from data.db import create_history_table
from interface.ui import demo

if __name__ == '__main__':

    create_history_table()

    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)


