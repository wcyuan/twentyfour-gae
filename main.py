from flask import Flask
import countdown
app = Flask(__name__)
app.config['DEBUG'] = True

# Note: We don't need to call run() since our application is embedded within
# the App Engine WSGI application server.


@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    
    (vals, target) = countdown.generate(
        4,
        24,
        num_large=0,
        replacement=True)
    
    results = tuple(countdown.countdown(
        vals, target,
        all_orders=True,
        all_subsets=False,
        use_pow=False))    
    return '<http><html><body><pre>{0}\n\n{1}</pre></body>'.format(
        ' '.join(str(v) for v in vals),
        '\n'.join(str(r) for r in results)
        )


@app.errorhandler(404)
def page_not_found(e):
    """Return a custom 404 error."""
    return 'Sorry, nothing at this URL.', 404
