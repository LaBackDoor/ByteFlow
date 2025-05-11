// Function to copy BibTeX citation to clipboard
function copyBibTeX() {
  var bibTexElement = document.querySelector(".bibtex-section pre code");
  var bibTexText = bibTexElement.innerText;
  navigator.clipboard.writeText(bibTexText);
  alert("BibTeX citation copied to clipboard!");
}

// Function to toggle dark mode
function toggleDarkMode() {
  document.body.classList.toggle("dark-mode");
  document.querySelector(".nav").classList.toggle("dark-mode");
  
  // Store user preference
  const isDarkMode = document.body.classList.contains("dark-mode");
  localStorage.setItem("darkMode", isDarkMode);
}

// Check for stored preferences when page loads
document.addEventListener("DOMContentLoaded", function() {
  const prefersDarkMode = localStorage.getItem("darkMode") === "true";
  if (prefersDarkMode) {
    document.body.classList.add("dark-mode");
    document.querySelector(".nav").classList.add("dark-mode");
  }
  
  // Handle scroll up button visibility
  window.onscroll = function() {
    const scrollUpBtn = document.getElementById("scrollUpBtn");
    if (document.body.scrollTop > 100 || document.documentElement.scrollTop > 100) {
      scrollUpBtn.style.display = "block";
    } else {
      scrollUpBtn.style.display = "none";
    }
  };
});

// Function to scroll to top
function scrollToTop() {
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// Add smooth scrolling for all anchor links
document.addEventListener("DOMContentLoaded", function() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      
      const targetId = this.getAttribute('href');
      const targetElement = document.querySelector(targetId);
      
      if (targetElement) {
        window.scrollTo({
          top: targetElement.offsetTop - 60, // Offset for the fixed nav
          behavior: 'smooth'
        });
      }
    });
  });
});
