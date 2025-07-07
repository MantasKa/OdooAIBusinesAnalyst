import streamlit as st
import os
import json
import hashlib
from typing import Dict, Optional, Tuple, List
from token_counter import TokenCounter

class LoginManager:
    """
    Manages user authentication for the Odoo GAP Analysis Assistant.
    Only invited users can login or register to the system.
    """

    def __init__(self, invited_users_file: str = "data/invited_users.json", token_data_file: str = "data/token_usage.json"):
        """
        Initialize the LoginManager.

        Args:
            invited_users_file: Path to the file containing invited users data
            token_data_file: Path to the file containing token usage data
        """
        # Convert to absolute path if it's a relative path
        if not os.path.isabs(invited_users_file):
            # For relative paths, use the current working directory instead of script directory
            # This ensures that "data/invited_users.json" resolves correctly when run from project root
            invited_users_file = os.path.join(os.getcwd(), invited_users_file)

        if not os.path.isabs(token_data_file):
            token_data_file = os.path.join(os.getcwd(), token_data_file)

        self.invited_users_file = invited_users_file
        self.token_counter = TokenCounter(token_data_file)
        self._ensure_data_directory()
        self._initialize_session_state()

    def _ensure_data_directory(self):
        """Ensure the data directory exists."""
        os.makedirs(os.path.dirname(self.invited_users_file), exist_ok=True)

        # Create invited users file if it doesn't exist
        if not os.path.exists(self.invited_users_file):
            self._save_invited_users([])

    def _initialize_session_state(self):
        """Initialize session state variables for authentication."""
        if 'user_logged_in' not in st.session_state:
            st.session_state.user_logged_in = False

        if 'current_user' not in st.session_state:
            st.session_state.current_user = None

    def _load_invited_users(self) -> List[Dict]:
        """Load the list of invited users from the file."""
        try:
            if os.path.exists(self.invited_users_file):
                with open(self.invited_users_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            st.error(f"Error loading invited users: {str(e)}")
            return []

    def _save_invited_users(self, users: List[Dict]):
        """Save the list of invited users to the file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.invited_users_file), exist_ok=True)

            with open(self.invited_users_file, 'w') as f:
                json.dump(users, f, indent=2)

            # Verify file was written
            if os.path.exists(self.invited_users_file):
                file_size = os.path.getsize(self.invited_users_file)

        except Exception as e:
            error_msg = f"Error saving invited users: {str(e)}"
            st.error(error_msg)
            st.write(f"**DEBUG:** Save failed with error: {error_msg}")

    def _hash_password(self, password: str) -> str:
        """Hash a password for secure storage."""
        return hashlib.sha256(password.encode()).hexdigest()

    def is_email_invited(self, email: str) -> bool:
        """Check if an email is in the invited users list."""
        invited_users = self._load_invited_users()
        return any(user.get('email', '').lower() == email.lower() for user in invited_users)

    def register_user(self, email: str, password: str, name: str) -> Tuple[bool, str]:
        """
        Register a new user if they are invited.

        Args:
            email: User's email
            password: User's password
            name: User's name

        Returns:
            Tuple of (success, message)
        """
        if not self.is_email_invited(email):
            return False, "This email is not invited to the system."

        invited_users = self._load_invited_users()

        # Check if user already exists
        for user in invited_users:
            if user.get('email', '').lower() == email.lower():
                if user.get('registered', False):
                    return False, "This email is already registered. Please login instead."

                # Update the invited user with registration info
                user['name'] = name
                user['password'] = self._hash_password(password)
                user['registered'] = True

                self._save_invited_users(invited_users)
                return True, "Registration successful! You can now login."

        return False, "An error occurred during registration."

    def login_user(self, email: str, password: str) -> Tuple[bool, str]:
        """
        Login a user with email and password.

        Args:
            email: User's email
            password: User's password

        Returns:
            Tuple of (success, message)
        """
        invited_users = self._load_invited_users()

        for user in invited_users:
            if user.get('email', '').lower() == email.lower():
                if not user.get('registered', False):
                    return False, "You need to register first."

                if user.get('password') == self._hash_password(password):
                    st.session_state.user_logged_in = True
                    st.session_state.current_user = {
                        'email': email,
                        'name': user.get('name', '')
                    }
                    return True, f"Welcome back, {user.get('name', '')}!"
                else:
                    return False, "Incorrect password."

        return False, "Email not found. Please check if you're invited."

    def logout_user(self):
        """Logout the current user."""
        st.session_state.user_logged_in = False
        st.session_state.current_user = None

    def is_logged_in(self) -> bool:
        """Check if a user is currently logged in."""
        return st.session_state.get('user_logged_in', False)

    def get_current_user(self) -> Optional[Dict]:
        """Get the currently logged in user."""
        return st.session_state.get('current_user')

    def add_invited_user(self, email: str) -> Tuple[bool, str]:
        """
        Add a new invited user to the system.

        Args:
            email: Email to invite

        Returns:
            Tuple of (success, message)
        """
        if not email or '@' not in email:
            return False, "Please provide a valid email address."

        try:
            invited_users = self._load_invited_users()


            # Check if already invited
            if any(user.get('email', '').lower() == email.lower() for user in
                   invited_users):
                return False, "This email is already invited."

            # Get current time
            import datetime
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Add new invited user
            new_user = {
                'email': email.lower(),
                'registered': False,
                'invited_at': current_time
            }

            invited_users.append(new_user)


            # Save the updated list
            self._save_invited_users(invited_users)

            # Verify the save was successful by reloading
            verified_users = self._load_invited_users()

            if any(user.get('email', '').lower() == email.lower() for user in
                   verified_users):
                return True, f"Successfully invited {email}."
            else:
                return False, f"Failed to save invitation for {email}. Please try again."

        except Exception as e:
            error_msg = f"An error occurred while inviting {email}: {str(e)}"
            st.error(error_msg)
            return False, error_msg

    def render_login_form(self):
        """Render the login form UI."""
        st.title("🔐 Login to Odoo GAP Analysis Assistant")

        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            with st.form("login_form"):
                st.subheader("Login")
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")

                submit_button = st.form_submit_button("Login")

                if submit_button:
                    if not email or not password:
                        st.error("Please fill in all fields.")
                    else:
                        success, message = self.login_user(email, password)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)

        with tab2:
            with st.form("register_form"):
                st.subheader("Register")
                st.info("Only invited users can register. Please enter your invited email address.")

                name = st.text_input("Full Name", key="register_name")
                email = st.text_input("Email", key="register_email")
                password = st.text_input("Password", type="password", key="register_password")
                confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm")

                submit_button = st.form_submit_button("Register")

                if submit_button:
                    if not name or not email or not password or not confirm_password:
                        st.error("Please fill in all fields.")
                    elif password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        success, message = self.register_user(email, password, name)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)

    def render_admin_panel(self):
        """Render the admin panel for managing invited users and viewing token usage."""
        # Create tabs for different admin sections
        tab1, tab2 = st.tabs(["👥 User Management", "🔢 Token Usage"])

        with tab1:
            st.title("👑 Admin Panel - Invite Management")

            # Add new invited user
            with st.form("invite_form"):
                st.subheader("Invite New User")
                email = st.text_input("Email to invite")
                submit_button = st.form_submit_button("Send Invitation")

                if submit_button:
                    if not email:
                        st.error("Please enter an email address.")
                    else:
                        success, message = self.add_invited_user(email)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)

            # List of invited users
            st.subheader("Invited Users")
            invited_users = self._load_invited_users()

            if not invited_users:
                st.info("No users have been invited yet.")
            else:
                for i, user in enumerate(invited_users):
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.write(f"**{user.get('email')}**")
                    with col2:
                        st.write("✅ Registered" if user.get('registered') else "⏳ Pending")
                    with col3:
                        if st.button("Remove", key=f"remove_{i}"):
                            invited_users.pop(i)
                            self._save_invited_users(invited_users)
                            st.success(f"Removed {user.get('email')}")
                            st.rerun()
                    st.divider()

        with tab2:
            st.title("🔢 Token Usage Statistics")

            # Get token usage data
            all_users_token_usage = self.token_counter.get_all_users_token_usage()
            total_token_usage = self.token_counter.get_total_token_usage()

            # Display total token usage
            st.subheader("Total Token Usage")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Input Tokens", f"{total_token_usage['input_tokens']:,}")
            with col2:
                st.metric("Total Output Tokens", f"{total_token_usage['output_tokens']:,}")

            st.divider()

            # Display token usage per user
            st.subheader("Token Usage per User")

            if not all_users_token_usage:
                st.info("No token usage data available yet.")
            else:
                # Create a selectbox to choose a user
                user_emails = list(all_users_token_usage.keys())
                selected_user = st.selectbox("Select User", user_emails)

                if selected_user:
                    user_data = all_users_token_usage[selected_user]

                    # Display user's total token usage
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Input Tokens", f"{user_data['total_input_tokens']:,}")
                    with col2:
                        st.metric("Total Output Tokens", f"{user_data['total_output_tokens']:,}")

                    # Display token usage per model
                    st.subheader(f"Model Usage for {selected_user}")

                    if not user_data['models']:
                        st.info("No model usage data available for this user.")
                    else:
                        # Create a table of model usage
                        model_data = []
                        for model_name, model_stats in user_data['models'].items():
                            model_data.append({
                                "Model": model_name,
                                "Input Tokens": f"{model_stats['input_tokens']:,}",
                                "Output Tokens": f"{model_stats['output_tokens']:,}",
                                "Total Tokens": f"{model_stats['input_tokens'] + model_stats['output_tokens']:,}"
                            })

                        # Display as a dataframe
                        st.dataframe(model_data)


def render_login_ui():
    """Main function to render the login UI."""
    login_manager = LoginManager()

    # If user is not logged in, show login form
    if not login_manager.is_logged_in():
        login_manager.render_login_form()
        return False, login_manager

    # If user is logged in, show logout option in sidebar
    current_user = login_manager.get_current_user()

    with st.sidebar:
        st.write(f"👤 Logged in as: **{current_user.get('name')}**")
        if st.button("Logout"):
            login_manager.logout_user()
            st.rerun()

    # Admin panel (for demonstration, we'll make the first registered user an admin)
    # In a real application, you would have proper admin roles
    invited_users = login_manager._load_invited_users()
    registered_users = [u for u in invited_users if u.get('registered', False)]

    if registered_users and registered_users[0].get('email') == current_user.get(
            'email'):
        # Add admin panel state to session
        if 'show_admin_panel' not in st.session_state:
            st.session_state.show_admin_panel = False

        if st.sidebar.button("Admin Panel"):
            st.session_state.show_admin_panel = not st.session_state.show_admin_panel
            st.rerun()

        # Show admin panel if toggled
        if st.session_state.show_admin_panel:
            login_manager.render_admin_panel()
            # Add a button to return to main app
            if st.button("← Back to Main App"):
                st.session_state.show_admin_panel = False
                st.rerun()
            return False, login_manager

    return True, login_manager

if __name__ == "__main__":
    # For testing the login form independently
    render_login_ui()
