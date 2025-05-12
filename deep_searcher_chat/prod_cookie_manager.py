import streamlit as st
import uuid
from typing import Optional

COOKIE_NAME = "my_anon_session_id"

def set_browser_cookie_if_missing():
    """
    Embeds a small JavaScript snippet to:
      1 Check if the cookie exists in document.cookie.
      2 If missing, generate a random UUID and set it in the cookie.
      3 Return the cookie value back to Streamlit via st.session_state.
    """
    if "client_cookie" not in st.session_state:
        # We'll run JS to create/read the cookie and store it in session_state
        st.session_state["client_cookie"] = str(uuid.uuid4())  # Default fallback
    
    cookie_manager_js = f"""
        <script>
        (function() {{
            var cookieName = "{COOKIE_NAME}";
            function getCookie(name) {{
                const value = "; " + document.cookie;
                const parts = value.split("; " + name + "=");
                if (parts.length === 2) return parts.pop().split(";").shift();
            }}

            var existing = getCookie(cookieName);
            if(!existing) {{
                // no cookie set => generate a random ID
                var newId = "{st.session_state['client_cookie']}";
                var d = new Date();
                d.setTime(d.getTime() + (30*24*60*60*1000)); // 30 days expiry
                var expires = "expires="+ d.toUTCString();
                document.cookie = cookieName + "=" + newId + ";" + expires + ";path=/";
                existing = newId;
            }}
            
            // Streamlit communication: 
            // If the cookie was found/created, pass it via Streamlit's 
            // sessionState using a custom callback:
            window.parent.postMessage({{"cookieValue": existing}}, "*");
        }})();
        </script>
    """
    st.markdown(cookie_manager_js, unsafe_allow_html=True)

def listen_for_cookie_from_js():
    """
    Listens for the posted message from the JS snippet containing the cookie value.
    We do this via Streamlit's built-in 'on_event' approach or an event in JS.
    In practice, we'll poll st.session_state['client_cookie'] on rerun or 
    use st.experimental_singleton approach. But here's a minimal example.
    """
    # We'll do an empty placeholder for new cookie data
    cookie_value = st.session_state.get("client_cookie", None)

    # Listen for posted messages in an HTML snippet:
    # In Streamlit, we can't truly handle postMessage easily. 
    # So a simpler approach is we rely on the user script to set st.session_state directly.
    # Or we can automatically run st.experimental_rerun from JS. 
    # For demonstration, let's keep it short and sweet.
    
    st.markdown("""
    <script>
    window.addEventListener("message", (event) => {
        if (event.data && event.data.cookieValue) {
            // Use Streamlit's prompt to set session state 
            // by writing to a hidden text input or something similar.
            const cookieVal = event.data.cookieValue;
            window.parent.postMessage({type: "streamlit:setSessionState", key: "client_cookie", value: cookieVal}, "*");
        }
    }, false);
    </script>
    """, unsafe_allow_html=True)

def get_session_id() -> Optional[str]:
    """
    Returns the session ID from st.session_state if set, else None.
    """
    return st.session_state.get("client_cookie", None)
